import cv2
import numpy as np

# 使用比较函数对轮廓进行排序
def contour_area(contour):
    return cv2.contourArea(contour)

def process_frame(frame):
    change_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_bound = np.array([100, 43, 46], dtype=np.uint8)
    upper_bound = np.array([124, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(change_frame, lower_bound, upper_bound)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.erode(mask, kernel, iterations=1)
    return mask



cap = cv2.VideoCapture(0)
find_first = 0
begin_sign = 0
while True:
    ret, frame = cap.read()
    processed_frame = process_frame(frame)
    contours, hierarchy = cv2.findContours(processed_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if begin_sign == 0:
        if find_first == 0:
            if len(contours) > 0:
                contours = sorted(contours, key=cv2.contourArea, reverse=True)
                new_contours = []
                for contour in contours:
                    (x, y), radius = cv2.minEnclosingCircle(contour)
                    center = (int(x), int(y))
                    # 去除掉上面和下面的噪点
                    if center[1] > 180 and center[1] < 320:
                        new_contours.append(contour)

                # 将新的列表赋值给 contours
                contours = new_contours
                if(len(contours) > 0):

                # 检查列表中第一个轮廓的面积是否大于 1300
                    if cv2.contourArea(contours[0]) > 1300:
                        print("找到了最大的蓝色挡板")
                        (x, y), radius = cv2.minEnclosingCircle(contours[0])
                        center = (int(x), int(y))
                        radius = int(radius)
                        cv2.circle(frame, center, radius, (0, 255, 0), 2)
                        find_first = 1
                    else:
                        print("没找到蓝色挡板")
        else:
            print("进入移开挡板的程序")
            if len(contours) > 0:
                contours = sorted(contours, key=cv2.contourArea, reverse=True)
                new_contours = []
                for contour in contours:
                    (x, y), radius = cv2.minEnclosingCircle(contour)
                    center = (int(x), int(y))
                    # 去除掉上面和下面的噪点
                    if center[1] > 180 and center[1] < 320:
                        new_contours.append(contour)

                contours = new_contours
                if(len(contours) > 0):
                    if cv2.contourArea(contours[0]) < 300:
                        begin_sign = 1
                        print("挡板移开")
                else:
                    begin_sign = 1
                    print("挡板移开")
    cv2.imshow('Original Frame', frame)
    cv2.imshow('Processed Frame', processed_frame)

    # 检测按键，如果按下Esc键则退出循环
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
