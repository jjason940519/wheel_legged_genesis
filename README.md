# wheel_legged_genesis
Reinforcement learning of wheel-legged robots based on Genesis  
## System Requirements  
Ubuntu 20.04/22.04/24.04  
## Hardware requirements  
NVIDIA/AMD GPU or CPU  
## Test & Development Platform  
1. i5 12400f +  Geforce RTX4070  
2. i7 12700kf + Radeon Rx7900xt
## Before running
Install Genesis:  
<https://github.com/Genesis-Embodied-AI/Genesis>  
install rsl-rl:    
`git clone https://github.com/leggedrobotics/rsl_rl`  
`cd rsl_rl && git checkout v1.0.2 && pip install -e .`  
install tensorboard:    
`pip install tensorboard`  
install taichi:  
`pip install taichi`  
install Imath:  
`pip install Imath`  
## Use
run:  
`git clone https://github.com/Albusgive/wheel_legged_genesis.git`  
`cd wheel_legged_genesis`  
train:  
`python locomotion/wheel_legged_train.py`  
test:  
`python locomotion/wheel_legged_eval.py`  
## Suggestion
When using NVIDIA GPUs, it is recommended that the genesis backend choose gpu or cuda    
When using AMD GPUs, it is recommended to select vulkan for the genesis backend  
## Demo of the effect    
[25赛季平衡底盘仿真](https://www.bilibili.com/video/BV1DUNHe7EjP/?share_source=copy_web>)  
[别平步了，要不双足吧](https://www.bilibili.com/video/BV1oSN8eUEXw/?share_source=copy_web>)   
[这是平步模式了](https://www.bilibili.com/video/BV1YoNDevENT/?share_source=copy_web>)
