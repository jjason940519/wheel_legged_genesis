import argparse
import os
import pickle
import shutil

from wheel_legged_env import WheelLeggedEnv
from rsl_rl.runners import OnPolicyRunner

import genesis as gs


def get_train_cfg(exp_name, max_iterations):

    train_cfg_dict = {
        "algorithm": {
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.01,
            "gamma": 0.995,
            "lam": 0.95,
            "learning_rate": 0.005,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 10,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "init_member_classes": {},
        "policy": {
            "activation": "elu",
            "actor_hidden_dims": [512, 256, 256, 128],
            "critic_hidden_dims": [512, 256, 256, 128],
            "init_noise_std": 1.5,
        },
        "runner": {
            "algorithm_class_name": "PPO",
            "checkpoint": -1,
            "experiment_name": exp_name,
            "load_run": -1,
            "log_interval": 1,
            "max_iterations": max_iterations,
            "num_steps_per_env": 24,
            "policy_class_name": "ActorCritic",
            "record_interval": -1,
            "resume": False,
            "resume_path": None,
            "run_name": "",
            "runner_class_name": "runner_class_name",
            "save_interval": 100,
        },
        "runner_class_name": "OnPolicyRunner",
        "seed": 1,
    }

    return train_cfg_dict


def get_cfgs():
    env_cfg = {
        "num_actions": 6,
        # joint/link names
        "default_joint_angles": {  # [rad]
            # "left_hip_joint":0.0,
            "left_thigh_joint": 0.0,
            "left_calf_joint": 0.0,
            # "right_hip_joint":0.0,
            "right_thigh_joint": 0.0,
            "right_calf_joint": 0.0,
            "left_wheel_joint": 0.0,
            "right_wheel_joint": 0.0,
        },
        "dof_names": [
            # "left_hip_joint",
            "left_thigh_joint",
            "left_calf_joint",
            # "right_hip_joint",
            "right_thigh_joint",
            "right_calf_joint",
            "left_wheel_joint",
            "right_wheel_joint",
        ],
        # lower upper
        "dof_limit": {
            # "left_hip_joint":[-0.31416, 0.31416],
            "left_thigh_joint": [-0.785399, 0.785399],
            "left_calf_joint": [0.0, 1.3963],
            # "right_hip_joint":[-0.31416, 0.31416],
            "right_thigh_joint": [-0.785399, 0.785399],
            "right_calf_joint": [0.0, 1.3963],
            "left_wheel_joint": [0.0, 0.0],
            "right_wheel_joint": [0.0, 0.0],
        },
        "safe_force": {
            # "left_hip_joint":23.6,
            "left_thigh_joint": 40.0,
            "left_calf_joint": 40.0,
            # "right_hip_joint":23.6,
            "right_thigh_joint": 40.0,
            "right_calf_joint": 40.0,
            "left_wheel_joint": 12.0,
            "right_wheel_joint": 12.0,
        },
        # PD
        "kp": 40.0,
        "kd": 5,
        # termination
        "termination_if_roll_greater_than": 10,  # degree
        "termination_if_pitch_greater_than": 10,
        "termination_if_base_height_greater_than": 0.10,
        "termination_base_height_time": 1.0,
        # base pose
        "base_init_pos": [0.0, 0.0, 0.13], #[0.0, 0.0, 0.3]
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 30.0,
        "resampling_time_s": 4.0,
        "joint_action_scale": 0.25,
        "wheel_action_scale": 0.25,
        "simulate_action_latency": True,
        "clip_actions": 100.0,
    }
    obs_cfg = {
        # num_obs = num_slice_obs + history_num * num_slice_obs
        "num_obs": 174, #在rsl-rl中使用的变量为num_obs表示state数量
        "num_slice_obs": 29,
        "history_length": 5,
        "obs_scales": {
            "lin_vel": 2.0,
            "lin_acc": 2.0,
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
            "height_measurements": 5.0,
        },
    }
    # 名字和奖励函数名一一对应
    reward_cfg = {
        "tracking_sigma": 0.25,
        "feet_height_target": 0.0,
        "reward_scales": {
            "tracking_lin_vel": 2.0,
            "tracking_ang_vel": 0.5,
            "tracking_base_height": 50.0,
            "lin_vel_z": -0.1,
            "joint_action_rate": -0.005,
            "wheel_action_rate": -0.0001,
            "similar_to_default": 0.0,
            "projected_gravity": 6,
            "similar_legged": 0.4,
        },
    }
    command_cfg = {
        "num_commands": 4,
        "lin_vel_x_range": [-3.0, 3.0],
        "lin_vel_y_range": [-0.0, 0.0],
        "ang_vel_range": [-1.0, 1.0],
        "height_target_range": [0.25 , 0.3],
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="wheel-legged-walking")
    parser.add_argument("-B", "--num_envs", type=int, default=4096)
    parser.add_argument("--max_iterations", type=int, default=5000)
    args = parser.parse_args()

    gs.init(logging_level="warning",backend=gs.vulkan)

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations)

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    env = WheelLeggedEnv(
        num_envs=args.num_envs, env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg, command_cfg=command_cfg
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device="cuda:0")

    pickle.dump(
        [env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg],
        open(f"{log_dir}/cfgs.pkl", "wb"),
    )

    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()

"""
# training
python examples/locomotion/go2_train.py
"""
