import genesis as gs
import numpy as np
import time
import cv2
gs.init(backend=gs.vulkan)

scene = gs.Scene(
    show_viewer = True,
    viewer_options = gs.options.ViewerOptions(
        res           = (1280, 960),
        camera_pos    = (3.5, 0.0, 2.5),
        camera_lookat = (0.0, 0.0, 0.5),
        camera_fov    = 40,
        max_FPS       = 60,
    ),
    vis_options = gs.options.VisOptions(
        show_world_frame = True,
        world_frame_size = 1.0,
        show_link_frame  = False,
        show_cameras     = False,
        plane_reflection = True,
        ambient_light    = (0.1, 0.1, 0.1),
    ),
    renderer=gs.renderers.Rasterizer(),
)

plane = scene.add_entity(
    gs.morphs.Plane(),
)


#高度字段行是是一个二维数组 一维数组数据为x轴数据 单位
# height_field = np.array([
#                          [0, 0, 0,0,0],
#                          [0, 0, 0,0,0],
#                          [0, 0, 0,0,0],
#                          [0, 0, 0,0,0], 
#                          [200, 200, 200,200,200], 
#                          [200, 200, 200,200,200],
#                          [200, 200, 200,200,200],
#                          [20, 20, 20,20,20],
#                          [12, 5, 13,34,54],
#                          [42, 40, 26,35,55],
#                          [15, 50, 21,36,56],
#                          [2, 30, 41,37,57],])
height_field = cv2.imread("/home/albusgive/wheel_legged_genesis/assets/terrain/png/jilin.png", cv2.IMREAD_GRAYSCALE)
height_field = cv2.resize(height_field,(125,100))   #建议100*100或者150*150
terrain = scene.add_entity(
        morph=gs.morphs.Terrain(
        pos = (0.5,0.5,0.0),
        height_field = height_field,
        horizontal_scale=0.1,  #水平缩放 (m) 建议0.1
        vertical_scale=0.005,   #垂直缩放 (m)
        ),
    )

robot = scene.add_entity(
    gs.morphs.URDF(file="assets/urdf/nz2/urdf/nz2.urdf",
    pos=(0.0, 0.0, 0.15)
    ),
    # gs.morphs.MJCF(file="assets/mjcf/nz/nz.xml",
    # pos=(0.0, 0.0, 0.265)
    # ),
    # vis_mode='collision'
)

cam = scene.add_camera(
    res    = (640, 480),
    pos    = (3.5, 0.0, 2.5),
    lookat = (0, 0, 0.5),
    fov    = 30,
    GUI    = False,
)
scene.build(n_envs=2)

jnt_names = [
    # "left_hip_joint",
    "left_thigh_joint",
    "left_calf_joint",
    # "right_hip_joint",
    "right_thigh_joint",
    "right_calf_joint",
    "left_wheel_joint",
    "right_wheel_joint",
]
dofs_idx = [robot.get_joint(name).dof_idx_local for name in jnt_names]
robot.set_dofs_kp(
    kp = np.array([20,20,20,20,40,40]),
    dofs_idx_local = dofs_idx,
)
robot.set_dofs_kv(
    kv = np.array([0.5,0.5,0.5,0.5,0.5,0.5]),
    dofs_idx_local = dofs_idx,
)
left_knee = robot.get_joint("left_calf_joint")

print(robot.n_links)
link = robot.get_link("base_link")
print(link.idx)
link = robot.get_link("left_wheel_link")
print(link.idx)
link = robot.get_link("right_wheel_link")
print(link.idx)

# 渲染rgb、深度、分割掩码和法线图
# rgb, depth, segmentation, normal = cam.render(rgb=True, depth=True, segmentation=True, normal=True)

# cam.start_recording()
import numpy as np

while True:
    # robot.control_dofs_position(
    #         np.array([0.0, 0.0, 0.0, 0.0, 10.0, 10.0]),
    #         dofs_idx,
    #     )
    scene.step()
    # print(robot.get_pos())
    # left_knee_pos = left_knee.get_pos()
    # print("left_knee_pos    ",left_knee_pos)
    # force = robot.get_links_net_contact_force()
    # dof_vel = robot.get_dofs_velocity()
    # print("dof_vel:",dof_vel)
    # time.sleep(0.1)
    cam.render()
# cam.stop_recording(save_to_filename='video.mp4', fps=60)