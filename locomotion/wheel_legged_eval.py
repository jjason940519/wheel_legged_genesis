import argparse
import os
import pickle

import torch
from wheel_legged_env import WheelLeggedEnv
from rsl_rl.runners import OnPolicyRunner

import genesis as gs # type: ignore

import sys
# 获取当前脚本所在的目录（sim2sim）
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取父级目录
parent_dir = os.path.dirname(current_dir)
# 将 parent_dir 添加到 sys.path，这样就能访问到 logs 和 utils 目录中的文件
sys.path.append(parent_dir)
from utils import gamepad
import copy
  
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="wheel-legged-walking")
    parser.add_argument("--ckpt", type=int, default=2700)
    args = parser.parse_args()
    

    gs.init(backend=gs.vulkan,logging_level="warning")
    gs.device="cuda:0"
    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg, curriculum_cfg, domain_rand_cfg, terrain_cfg, train_cfg = pickle.load(open(f"logs/{args.exp_name}/cfgs.pkl", "rb"))
    terrain_cfg["terrain"] = True
    terrain_cfg["eval"] = "agent_eval_gym" #agent_eval_gym/agent_train_gym/circular
    # env_cfg["kp"] = 40
    env = WheelLeggedEnv(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        curriculum_cfg=curriculum_cfg,
        domain_rand_cfg=domain_rand_cfg,
        terrain_cfg=terrain_cfg,
        show_viewer=True,
        train_mode=False
    )
    print(reward_cfg)
    runner = OnPolicyRunner(env, train_cfg, log_dir, device="cuda:0")
    resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device="cuda:0")
    #jit
    model = copy.deepcopy(runner.alg.actor_critic.actor).to('cpu')
    torch.jit.script(model).save(log_dir+"/policy.pt")
    # 加载模型进行测试
    print("\n--- 模型加载测试 ---")
    try:
        loaded_policy = torch.jit.load(log_dir + "/policy.pt")
        loaded_policy.eval() # 设置为评估模式
        loaded_policy.to('cuda')
        print("模型加载成功!")
    except Exception as e:
        print(f"模型加载失败: {e}")
        exit()
    print("gs.tc_float ",gs.tc_float)
    obs, _ = env.reset()
    pad = gamepad.control_gamepad(command_cfg,[1.0,1.0,3.14,0.05])
    with torch.no_grad():
        while True:
            # actions = policy(obs)
            actions = loaded_policy(obs)
            print(actions)
            # data_list = [[-0.5,0.7,-0.5,0.7,0.0,0.0]]
            # actions = torch.tensor(data_list,device='cuda') 
            obs, _, rews, dones, infos = env.step(actions)
            comands,reset_flag = pad.get_commands()
            env.set_commands(0,comands)
            if reset_flag:
                env.reset()
            


if __name__ == "__main__":
    main()

"""
# evaluation
python examples/locomotion/go2_eval.py -e go2-walking -v --ckpt 100
"""
