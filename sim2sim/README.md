# Sim2Sim
**Two migration methods： CPP and Python, are provided to Mujoco**
**prepare**
You need to change the absolute path in sim2sim/scence.xml   
修改sim2sim/scence.xml中绝对路径部分  
## CPP
### Before running
```
sudo apt-get install parcellite  
sudo apt-get install libudev-dev  
sudo apt-get install joystick  
```
you need a gamepad
### run
1. Download and unzip libtorch.  
2. Download and install mujoco to /opt.  
3. Modify the TORCH_FOLDER path in the CMakeLists.txt file to your libtorch/share/cmake.  
4. `cd build.`  
5. `cmake ..`  
6. `make.`  
7. `./gs2mj`  
## python
### Before running
pip install mujoco
### run
in sim2sim `python gs2mj.py`
### note  
When the joint angle in the mujoco reaches +-300, it is forced to lock again, and do not send a control command all the time during the test  
mujoco中关节角度在达到+-300的时候回强制锁死，测试的过程中不要一直发送一个控制指令  
