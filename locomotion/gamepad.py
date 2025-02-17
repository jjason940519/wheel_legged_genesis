import pygame
import numpy as np
class control_gamepad:
    def __init__(self,command_cfg,command_scale=None):
        pygame.init()
        pygame.joystick.init()
        # 获取连接的游戏手柄数量
        joystick_count = pygame.joystick.get_count()
        if joystick_count == 0:
            print("no gamepad")
        else:
            # 选择第一个手柄
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
            # print(f"link gamepad: {self.joystick.get_name()}")
        self.num_commands = command_cfg["num_commands"]
        self.command_cfg = command_cfg
        self.commands = np.zeros(self.num_commands)
        self.command_scale = command_scale
        if self.command_scale is None:
            self.command_scale = [1.0, 1.0, 1.0]
    
    def get_commands(self):
        pygame.event.pump()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.JOYBUTTONDOWN:
                print(f"按钮 {event.button} 被按下。")
            elif event.type == pygame.JOYBUTTONUP:
                print(f"按钮 {event.button} 被释放。")
            elif event.type == pygame.JOYAXISMOTION:
                # print(f"轴 {event.axis},{event.value}")
                match event.axis:
                    case 1: #ly 前正后负
                        self.commands[0] = -event.value * self.command_scale[0]
                    case 2: #rx 左正右负
                        self.commands[2] = -event.value * self.command_scale[2]
                    case 3: #lt 增加身高
                        self.commands[3] += (event.value+1)/100
                    case 4: #ry 减少身高
                        self.commands[3] -= (event.value+1)/100
        self.commands_clip()
        return self.commands
    
    def commands_clip(self):
        # lin_vel_x
        if self.commands[0] < self.command_cfg["lin_vel_x_range"][0] * self.command_scale[0]:
            self.commands[0] = self.command_cfg["lin_vel_x_range"][0] * self.command_scale[0]
        elif self.commands[0] > self.command_cfg["lin_vel_x_range"][1] * self.command_scale[0]:
            self.commands[0] = self.command_cfg["lin_vel_x_range"][1] * self.command_scale[0]

        #lin_vel_y
        if self.commands[1] < self.command_cfg["lin_vel_y_range"][0] * self.command_scale[1]:
            self.commands[1] = self.command_cfg["lin_vel_y_range"][0] * self.command_scale[1]
        elif self.commands[1] > self.command_cfg["lin_vel_y_range"][1] * self.command_scale[1]:
            self.commands[1] = self.command_cfg["lin_vel_y_range"][1] * self.command_scale[1]

        #ang_vel
        if self.commands[2] < self.command_cfg["ang_vel_range"][0] * self.command_scale[2]:
            self.commands[2] = self.command_cfg["ang_vel_range"][0] * self.command_scale[2]
        elif self.commands[2] > self.command_cfg["ang_vel_range"][1] * self.command_scale[2]:
            self.commands[2] = self.command_cfg["ang_vel_range"][1] * self.command_scale[2]

        #base_heigh
        if self.commands[3] < self.command_cfg["height_target_range"][0]:
            self.commands[3] = self.command_cfg["height_target_range"][0]
        elif self.commands[3] > self.command_cfg["height_target_range"][1]:
            self.commands[3] = self.command_cfg["height_target_range"][1]