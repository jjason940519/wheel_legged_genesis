import cv2
import numpy as np
import math

img_size = 80
horizontal_scale = 0.1
vertical_scale = 0.001   #垂直缩放(单像素高度)
slope = np.zeros((img_size, img_size), dtype=np.uint8)

center = img_size//2
angle_degrees = 17
thickness = 1
max_height = math.tan(math.radians(angle_degrees))*(img_size*horizontal_scale) #现实高度
flag = 1 #1上坡 -1下坡 0维持
keep = 2
cnt = 0
color = vertical_scale*3
for r in range(3,img_size):
    match flag:
        case 1:
            color += (1/img_size)*max_height / vertical_scale
        case -1:
            color -= (1/img_size)*max_height / vertical_scale
        case 0:
            cnt += 1
    if cnt>keep:
        if color==255:
            flag=-1
        elif color==vertical_scale*3:
            flag=1
        cnt=0
    else:
        if color>255:
            color = 255
            flag=0
        elif color < 0:
            color=vertical_scale*3
            flag=0
    # cv2.circle(slope, (center,center), r, color, thickness)
    
    x1 = int(center - r / 2)
    y1 = int(center - r / 2)
    x2 = int(center + r / 2)
    y2 = int(center + r / 2)
    cv2.rectangle(slope, (x1, y1), (x2, y2), color, thickness)

print(f"real size:{img_size*horizontal_scale} m")
print(f"center pos:({center},{center},{0})")
# 保存图像
cv2.imwrite("./png/slope.png", slope)
