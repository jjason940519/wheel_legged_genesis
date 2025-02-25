# sim2sim
**Two migration methods： CPP and Python, are provided to Mujoco**
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
