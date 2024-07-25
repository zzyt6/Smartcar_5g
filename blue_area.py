import cv2
import numpy as np

def find_blue_contours(image_path):
    # 读取图像
    image = cv2.imread(image_path)

    # 将图像转换为HSV颜色空间
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 定义蓝色的HSV范围
    lower_blue = np.array([100, 43, 46])
    upper_blue = np.array([124, 255, 255])

    # 创建一个蓝色掩码
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # 执行形态学操作，可以根据实际情况调整参数
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 寻找轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 仅保留y轴在210-320范围内的轮廓
    filtered_contours = [contour for contour in contours if 95 <= cv2.boundingRect(contour)[1] <= 165]

    # 在图像上画出轮廓，并写出大小
    for i, contour in enumerate(filtered_contours):
        area = cv2.contourArea(contour)
        cv2.drawContours(image, [contour], 0, (0, 255, 0), 2)
        cv2.putText(image, f"Contour {i + 1}: {area:.2f}", (10, 30 * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # 显示图像
    cv2.imshow("Blue Contours", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    image_path = "/home/pi/test_photo/blue_zhuitong8.jpg"  # 替换为你的图像文件路径
    find_blue_contours(image_path)
