import cv2
import numpy as np

img_size = 80
horizontal_scale = 0.1
vertical_scale = 0.001   #垂直缩放(单像素高度)
#缩放参考 水平0.1 垂直0.02
mean_noise_hight = 0.05 #噪声平均高度 m
rugged = np.random.randint(0, mean_noise_hight/vertical_scale, (img_size, img_size), dtype=np.uint8)

#平地位置
center = img_size//2
center_hight = 80
plane_size = 4
rugged[center-plane_size:center+plane_size, center-plane_size:center+plane_size] = center_hight

#缓坡
top_left = 0
color = 0
thickness = 1
for i in range(0,5):
    color +=i*5
    cv2.rectangle(rugged, (top_left+i,top_left+i), (img_size-i,img_size-i), color, thickness)

print(f"horizontal_scale:{horizontal_scale} vertical_scale:{vertical_scale}")
print(f"real size:{img_size*horizontal_scale} m")
print(f"center pos:({center*horizontal_scale},{center*horizontal_scale},{center_hight*vertical_scale})")
# 保存图像
cv2.imwrite("./png/rugged.png", rugged)
