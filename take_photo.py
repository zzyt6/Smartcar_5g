import cv2
import os
import numpy as np



def undistort(frame):
    k1, k2, p1, p2, k3 = -0.340032906324299, 0.101344757327394,0.0,0.0, 0.0

    # 相机坐标系到像素坐标系的�?换矩�?
    k = np.array([
        [162.063205442089, 0, 154.707845362265],
        [0, 162.326264903804,129.914361509615],
        [0, 0, 1]
    ])
    # 畸变系数
    d = np.array([
        k1, k2, p1, p2, k3
    ])
    height, weight = frame.shape[:2]
    mapx, mapy = cv2.initUndistortRectifyMap(k, d, None, k, (weight, height), 5)

    # 返回�?正好的图形变�?
    return cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)

# 打开摄像�?
cap = cv2.VideoCapture(0)  # 0表示默�?�摄像头，�?�果有�?�个摄像头，�?以尝试不同的数字来选择不同的摄像头
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)  #设置宽度
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)  #设置长度
# 检查摄像头�?否成功打开
if not cap.isOpened():
    print("无法打开摄像�?.")
else:
    while True:
        # 捕捉一�?
        ret, frame = cap.read()

        if not ret:
            print("无法捕捉�?.")
            break
        frame = undistort(frame)
        # 显示当前�?
        cv2.imshow('Press "q" to Capture and Quit', frame)

        # 等待用户按下�?�?
        key = cv2.waitKey(1)

        # 如果用户按下 "q" �?，拍照并保存，然后退出程�?
        if key & 0xFF == ord('q'):
            # 构造完整的文件�?�?
            photo_path = os.path.join("test_photo", '40.jpg')
            cv2.imwrite(photo_path, frame)
            print(f"已保存照片为 {photo_path}")
            break

    # 释放摄像�?
    cap.release()

# 关闭所有窗�?
cv2.destroyAllWindows()
