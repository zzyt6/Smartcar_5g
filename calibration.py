# 摄像头畸变矫正函数，输入待矫正的图形变量
import cv2
import os
import numpy as np

def undistort(frame):
    k1, k2, p1, p2, k3 = -0.287246515012261, 0.066176222325459, 0.005615032474715,0.003425003902561, 0.0

    # 相机坐标系到像素坐标系的转换矩阵
    k = np.array([
        [3.111337497474041e+02, -2.333471935388314, 2.915941445374422e+02],
        [0, 3.109853062871910e+02, 2.473500696130221e+02],
        [0, 0, 1]
    ])
    # 畸变系数
    d = np.array([
        k1, k2, p1, p2, k3
    ])
    height, weight = frame.shape[:2]
    mapx, mapy = cv2.initUndistortRectifyMap(k, d, None, k, (weight, height), 5)

    # 返回矫正好的图形变量
    return cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
# 打开摄像头
cap = cv2.VideoCapture(0)  # 0表示默认摄像头，如果有多个摄像头，可以尝试不同的数字来选择不同的摄像头

if not cap.isOpened():
    print("无法打开摄像头.")
else:
    while True:
        # 捕捉一帧
        ret, frame = cap.read()

        if not ret:
            print("无法捕捉帧.")
            break
        frame = undistort(frame)
        # 显示当前帧
        cv2.imshow('Press "q" to Capture and Quit', frame)

        # 等待用户按下键盘
        key = cv2.waitKey(1)

        # # 如果用户按下 "q" 键，拍照并保存，然后退出程序
        # if key & 0xFF == ord('q'):
        #     # 构造完整的文件路径
        #     photo_path = os.path.join("test_photo", '30.jpg')
        #     cv2.imwrite(photo_path, frame)
        #     print(f"已保存照片为 {photo_path}")
        #     break

    # 释放摄像头
    cap.release()

# 关闭所有窗口
cv2.destroyAllWindows()

