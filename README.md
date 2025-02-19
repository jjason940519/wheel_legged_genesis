# wheel_legged_genesis
Reinforcement learning of wheel-legged robots based on Genesis  
## System Requirements  
Ubuntu 20.04/22.04/24.04  
python >= 3.10
## Hardware requirements  
NVIDIA/AMD GPU or CPU  
## must（必须）
**Use the main branch of Genesis to install it locally, and you cannot use Genesis 0.2.1 Release**  
**不能使用pip安装genesis的0.2.1版本，使用main分支的本地安装，因为有API不用**  
## Test & Development Platform  
1. i5 12400f +  Geforce RTX4070  
2. i7 12700kf + Radeon Rx7900xt
## Before running
### 1. Clone repo
run:  
```
git clone https://github.com/Albusgive/wheel_legged_genesis.git
cd wheel_legged_genesis
```

### 2. install deps
#### use pdm install
Install pdm, <https://pdm-project.org/en/latest/#installation>, then run
```
pdm install
```

#### or manual install
Install Genesis:  
<https://github.com/Genesis-Embodied-AI/Genesis>  
install tensorboard:    
`pip install tensorboard`  
`pip install pygame`   

install rsl-rl:    
`cd rsl_rl && pip install -e .`  

## Use
### use pdm
test:  
`pdm run locomotion/wheel_legged_eval.py`  
train:  
`pdm run locomotion/wheel_legged_train.py`  

### or manual
test:  
`python locomotion/wheel_legged_eval.py`  
train:  
`python locomotion/wheel_legged_train.py`  

### gamepad  
|key|function|
|---|--------|
|LY|lin_vel|
|RX|ang_vel|
|LT|height_up|
|RT|height_down| 
## Suggestion
When using NVIDIA GPUs, it is recommended that the genesis backend choose gpu or cuda    
When using AMD GPUs, it is recommended to select vulkan for the genesis backend  

## Demo of the effect    
[25赛季平衡底盘仿真](https://www.bilibili.com/video/BV1DUNHe7EjP/?share_source=copy_web>)  
[别平步了，要不双足吧](https://www.bilibili.com/video/BV1oSN8eUEXw/?share_source=copy_web>)   
[这是平步模式了](https://www.bilibili.com/video/BV1YoNDevENT/?share_source=copy_web>)    
[【开源】genesis双轮足机器人强化学习（基础版）](https://www.bilibili.com/video/BV14eKKeiEJB/?share_source=copy_web) 
## TODO:
**terrain：**   
Rough roads, uphill and downhill slopes, and stairs
**more dof:**
left_hip and right_hip
技术交流：  
![微信图片_20250214171412_3](https://github.com/user-attachments/assets/958ad5c0-b76e-4446-ba15-1bb4d1ac0383)  
