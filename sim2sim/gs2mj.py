import torch
import mujoco
import mujoco.viewer
import time
m = mujoco.MjModel.from_xml_path('./scence.xml')
d = mujoco.MjData(m)

def get_sensor_data(sensor_name):
    sensor_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
    data_pos = 0
    for i in range(sensor_id):
        data_pos += m.sensor_dim[i]
    for i in range(m.sensor_dim[sensor_id]):
        print(d.sensordata[data_pos + i])

with mujoco.viewer.launch_passive(m, d) as viewer:
    while viewer.is_running():
        step_start = time.time()
        mujoco.mj_step(m, d)
        # 更新渲染
        viewer.sync()
        # 同步时间
        time_until_next_step = m.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
