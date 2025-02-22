import sys
import os
import torch
import mujoco
import mujoco.viewer
import time
import argparse
import pickle
import numpy as np

# 加载 mujoco 模型
m = mujoco.MjModel.from_xml_path('sim2sim/scence.xml')
d = mujoco.MjData(m)

# 获取当前脚本所在的目录（sim2sim）
current_dir = os.path.dirname(os.path.abspath(__file__))

# 获取父级目录
parent_dir = os.path.dirname(current_dir)

# 将 parent_dir 添加到 sys.path，这样就能访问到 logs 和 utils 目录中的文件
sys.path.append(parent_dir)

# 导入 utils 中的 gamepad 模块
from utils import gamepad


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_sensor_data(sensor_name):
    # Find sensor ID based on sensor name
    sensor_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
    data_pos = 0
    for i in range(sensor_id):
        data_pos += m.sensor_dim[i]
    # Assuming `m.sensor_data` contains the raw data, and we're extracting the right slice
    sensor_data = d.sensordata[data_pos:data_pos + m.sensor_dim[sensor_id]]
    # Convert the data to a tensor and return
    return torch.tensor(sensor_data, device=device, dtype=torch.float32)

def world2self(quat, v):
    q_w = quat[0] 
    q_vec = quat[1:] 
    v_vec = torch.tensor(v, device=device,dtype=torch.float32)
    a = v_vec * (2.0 * q_w**2 - 1.0)
    b = torch.cross(q_vec, v_vec) * q_w * 2.0
    c = q_vec * torch.dot(q_vec, v_vec) * 2.0
    result = a - b + c
    return result.to(device)

def get_obs(env_cfg, obs_scales, actions, commands=[0.0, 0.0, 0.0, 0.22]):
    commands_scale = torch.tensor(
        [obs_scales["lin_vel"], obs_scales["lin_vel"], 
         obs_scales["ang_vel"], obs_scales["height_measurements"]], device=device, dtype=torch.float32)
    base_quat = get_sensor_data("orientation")
    gravity = [0.0, 0.0, -1.0]
    projected_gravity = world2self(base_quat,torch.tensor(gravity, device=device, dtype=torch.float32))
    base_lin_vel = world2self(base_quat,get_sensor_data("base_lin_vel"))
    base_ang_vel = world2self(base_quat,get_sensor_data("base_ang_vel"))
    dof_pos = torch.zeros(env_cfg["num_actions"], device=device, dtype=torch.float32)
    for i, dof_name in enumerate(env_cfg["dof_names"]):
        dof_pos[i] = get_sensor_data(dof_name+"_p")
        if i==3:
            break
    dof_vel = torch.zeros(env_cfg["num_actions"], device=device, dtype=torch.float32)
    for i, dof_name in enumerate(env_cfg["dof_names"]):
        dof_vel[i] = get_sensor_data(dof_name+"_v")
    
    cmds = torch.tensor(commands, device=device, dtype=torch.float32)
    
    default_dof_pos = torch.tensor(
            [env_cfg["default_joint_angles"][name] for name in env_cfg["dof_names"]],
            device=device,
            dtype=torch.float32)

    return torch.cat(
        [
            base_lin_vel * obs_scales["lin_vel"],  # 3
            base_ang_vel * obs_scales["ang_vel"],  # 3
            projected_gravity,  # 3
            cmds * commands_scale,  # 4
            (dof_pos[0:4] - default_dof_pos[0:4]) * obs_scales["dof_pos"],  # 4
            dof_vel * obs_scales["dof_vel"],  # 6
            actions,  # 6
        ],
        axis=-1,
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="wheel-legged-walking")
    parser.add_argument("--ckpt", type=int, default=2700)
    args = parser.parse_args()

    # 拼接到 logs 文件夹的路径
    log_dir = os.path.join(parent_dir, 'logs', args.exp_name)
    cfg_path = os.path.join(log_dir, 'cfgs.pkl')

    # 读取配置文件
    if os.path.exists(cfg_path):
        print("文件存在:", cfg_path)
        env_cfg, obs_cfg, reward_cfg, command_cfg, curriculum_cfg, domain_rand_cfg, terrain_cfg, train_cfg = pickle.load(open(cfg_path, "rb"))
    else:
        print("文件不存在:", cfg_path)
        exit()

    # 加载游戏控制器
    pad = gamepad.control_gamepad(command_cfg, [1.0, 1.0, 3.14, 0.05])
    commands, reset_flag = pad.get_commands()

    # 加载模型
    try:
        loaded_policy = torch.jit.load(os.path.join(log_dir, "policy.pt"))
        loaded_policy.eval()  # 设置为评估模式
        loaded_policy.to('cuda')
        print("模型加载成功!")
    except Exception as e:
        print(f"模型加载失败: {e}")
        exit()
    
    # 初始化观察数据
    history_obs_buf = torch.zeros((obs_cfg["history_length"], obs_cfg["num_slice_obs"]), device=device, dtype=torch.float32)
    slice_obs_buf = torch.zeros(obs_cfg["num_slice_obs"], device=device, dtype=torch.float32)
    obs_buf = torch.zeros((obs_cfg["num_obs"]), device=device, dtype=torch.float32)

    # 启动 mujoco 渲染
    with mujoco.viewer.launch_passive(m, d) as viewer:
        while viewer.is_running():
            actions = loaded_policy(obs_buf)
            slice_obs_buf = get_obs(env_cfg=env_cfg, obs_scales=obs_cfg["obs_scales"], actions=actions, commands=commands)
            slice_obs_buf = slice_obs_buf.unsqueeze(0)
            obs_buf = torch.cat([history_obs_buf, slice_obs_buf], dim=0).view(-1)

            # 更新历史缓冲区
            if obs_cfg["history_length"] > 1:
                history_obs_buf[:-1, :] = history_obs_buf[1:, :].clone()  # 移位操作
            history_obs_buf[-1, :] = slice_obs_buf 

            # 更新动作
            act = actions.detach().cpu().numpy()
            for i in range(env_cfg["num_actions"]):
                d.ctrl[i] = act[i]

            # 获取控制命令
            commands, reset_flag = pad.get_commands()
            if reset_flag:
                mujoco.mj_resetData(m, d)

            # 执行一步模拟
            step_start = time.time()
            for i in range(5):
                mujoco.mj_step(m, d)
            # 更新渲染
            viewer.sync()
            # 同步时间
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()
