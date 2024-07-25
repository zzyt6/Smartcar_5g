//======================================================================================================================================================================================#include <iostream>
#include <cstdlib>
#include <unistd.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdlib.h>
#include <opencv4/opencv2/core/core.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/imgproc/imgproc_c.h>
#include <string>
#include <pigpio.h>
#include <opencv2/opencv.hpp>
#include <thread>
#include <vector>
#include <chrono>

#define Usage()                                            \
    {                                                      \
        std::cerr << "usage: ./showpic FILE" << std::endl; \
    }
using namespace std;
using namespace cv;
cv::Mat openedImage;
Mat bin_image2;
Mat frame;
Mat img;
Mat bin_image;
Mat change_frame;
Mat mask;
Mat final_img;
Mat white_mask;
Mat hsv_image;
Mat delite_frame;
Mat binaryImage;
Mat grayImage;
vector<Point> contours;
vector<vector<Point>> contours_all;
int number1 = 0;
int number2 = 0;
int heighest = 100;
int count_change = 0;
int begin_sign = 1;
int find_first = 0;
int flag_yellow_finish = 0;
int flag_cross_finish = 0;
int cross;
int now_blue_max;
int flag_cross = 0;
int line_num = 0;
int bluearea_min = 50;
int bluearea_max = 1000;
int count_count2 = 0;
int patching_line1[2] = {1, 0};
int patching_line2[2] = {0, 1};
int try_patching_line = 1;
int yellow_flag = 0;
int sigmaCenter;
int speedlow = 11500;
int speedhigh = 12000;
int motorSpeed;
int counterShift = 0;
int last_blue = 0;
int blue_ver_mid;
int times4 = 0;
Mat binaryImage_1;
Point end_point;
Point vertebral_barrel_Point;
float kp = 1.0;
float kd = 2.0;
float servo_pwm_now;
float servo_pwm_yanzhen;
int error_first = 0;
int last_error = 0;
float servo_pwm_diff;
float servo_pwm;
std::vector<cv::Point> left_line;
std::vector<cv::Point> right_line;
std::vector<cv::Point> mid;
std::vector<cv::Point> left_line_blue;
std::vector<cv::Point> right_line_blue;
std::vector<cv::Point> mid_blue;
const int servo_pin = 12;
const float servo_pwm_range = 10000;
const float servo_pwm_frequency = 50;
const float servo_pwm_duty_cycle_unlock = 720;

const int motor_pin = 13;
const float motor_pwm_range = 40000.0;
const float motor_pwm_frequency = 200.0;
const float motor_pwm_duty_cycle_unlock = 10000.0;

cv::Mat drawWhiteLine(const cv::Mat &binaryImage, cv::Point start, cv::Point end, int lineWidth)
{
    cv::Mat resultImage = binaryImage.clone();

    int x1 = start.x, y1 = start.y;
    int x2 = end.x, y2 = end.y;

    if (x1 == x2)
    {
        for (int y = std::min(y1, y2); y <= std::max(y1, y2); ++y)
        {
            for (int i = -lineWidth / 2; i <= lineWidth / 2; ++i)
            {
                resultImage.at<uchar>(cv::Point(x1 + i, y)) = 255;
            }
        }
    }
    else
    {
        double slope = static_cast<double>(y2 - y1) / (x2 - x1);
        double intercept = y1 - slope * x1;

        for (int x = std::min(x1, x2); x <= std::max(x1, x2); ++x)
        {
            int y = static_cast<int>(slope * x + intercept);
            for (int i = -lineWidth / 2; i <= lineWidth / 2; ++i)
            {
                int newY = std::max(0, std::min(y + i, resultImage.rows - 1));
                resultImage.at<uchar>(cv::Point(x, newY)) = 255;
            }
        }
    }

    return resultImage;
}
void servo_motor_pwmInit(void)
{
    if (gpioInitialise() < 0)
    {
        std::cout << "GPIO failed" << std::endl;
        return;
    }
    else
        std::cout << "GPIO ok" << std::endl;
    gpioSetMode(servo_pin, PI_OUTPUT);
    gpioSetPWMfrequency(servo_pin, servo_pwm_frequency);
    gpioSetPWMrange(servo_pin, servo_pwm_range);
    gpioPWM(servo_pin, servo_pwm_duty_cycle_unlock);

    gpioSetMode(motor_pin, PI_OUTPUT);
    gpioSetPWMfrequency(motor_pin, motor_pwm_frequency);
    gpioSetPWMrange(motor_pin, motor_pwm_range);
    gpioPWM(motor_pin, motor_pwm_duty_cycle_unlock);
}

cv::Mat undistort(const cv::Mat &frame)
{
    double k1 = -0.340032906324299;
    double k2 = 0.101344757327394;
    double p1 = 0.0;
    double p2 = 0.0;
    double k3 = 0.0;

    cv::Mat K = (cv::Mat_<double>(3, 3) << 162.063205442089, 0.0, 154.707845362265,
                 0.0, 162.326264903804, 129.914361509615,
                 0.0, 0.0, 1.0);

    cv::Mat D = (cv::Mat_<double>(1, 5) << k1, k2, p1, p2, k3);
    cv::Mat mapx, mapy;
    cv::Mat undistortedFrame;

    cv::initUndistortRectifyMap(K, D, cv::Mat(), K, frame.size(), CV_32FC1, mapx, mapy);
    cv::remap(frame, undistortedFrame, mapx, mapy, cv::INTER_LINEAR);

    return undistortedFrame;
}
Mat customEqualizeHist(const Mat &inputImage, float alpha)
{
    Mat enhancedImage;
    equalizeHist(inputImage, enhancedImage);

    // 减弱对比度增强的效果
    return alpha * enhancedImage + (1 - alpha) * inputImage;
}
Mat ImagePreprocessing(const cv::Mat &frame_a)
{
    int width = 320;
    int height = 240;

    binaryImage_1 = cv::Mat::zeros(height, width, CV_8U);
    //--------------------------------------------这一段代码中包含了canny算子和一些图像处理的方案--------------------------------------------------
    // cv::Mat grayImage;
    // cv::cvtColor(frame, grayImage, cv::COLOR_BGR2GRAY);
    // cv::Mat edgeImage;
    // cv::Canny(grayImage, edgeImage, 50, 150);
    // vector<Vec4i> lines;
    // HoughLinesP(edgeImage, lines, 1, CV_PI / 180, 180, 100, 10);

    // for (size_t i = 0; i < lines.size(); i++) {
    //     Vec4i l = lines[i];
    //     line(binaryImage, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(255), 4);
    // }
    // // GaussianBlur(frame, img, cv::Size(7, 7), 0);
    // cvtColor(img, hsv_image, cv::COLOR_BGR2HSV);
    // // Scalar lower_white(10, 43, 46);
    // // Scalar upper_white(180, 255, 255);
    // // inRange(hsv_image, lower_white, upper_white, white_mask);
    // // bitwise_not(white_mask, final_img);
    // // cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1, 1));

    // // cv::morphologyEx(final_img, delite_frame, cv::MORPH_OPEN, kernel);
    // Mat img_edges;
    // Canny(frame, img_edges, 100, 150);
    // cv::Mat blurred;
    // cv::Mat edges;
    // cv::GaussianBlur(frame_a, blurred, cv::Size(5, 5), 0);
    // int roiYStart = 100;
    // int roiYEnd = 155;
    // // 创建感兴趣区域（ROI）
    // cv::Mat roiImage = blurred(cv::Range(roiYStart, roiYEnd), cv::Range(0, frame_a.cols));

    // // Canny边缘检测
    // cv::Canny(roiImage, edges, 100, 150);

    // // 创建全黑图像
    // cv::Mat blackImage = cv::Mat::zeros(frame_a.size(), CV_8UC1);

    // // 将Canny边缘叠加到全黑图像的感兴趣区域中
    // edges.copyTo(blackImage(cv::Range(roiYStart, roiYEnd), cv::Range(0, frame_a.cols)));
    // imshow("canny1",edges);
    // imshow("canny2",blackImage);
    //----------------------------------------------------------------------------------------------------------------
    //-------------------------------------------------赛道条件好的情况-------------------------------------------------
    // int kernelSize = 5;
    // double sigma = 1.0;
    // cv::Mat blurredImage;
    // cv::GaussianBlur(frame_a, blurredImage, cv::Size(kernelSize, kernelSize), sigma);
    // cv::Mat grad_x, grad_y;
    // cv::Mat sobel_x,sobel_y;
    // cv::Mat abs_grad_x, abs_grad_y;
    // int xWeight = 1;
    // int yWeight = 1;
    // cv::Sobel(blurredImage, grad_x, CV_16S, 1, 0, 3, xWeight);
    // cv::Sobel(blurredImage, grad_y, CV_16S, 0, 1, 3, yWeight);
    // cv::convertScaleAbs(grad_x, abs_grad_x);
    // cv::convertScaleAbs(grad_y, abs_grad_y);
    // cv::Mat edges;
    // cv::addWeighted(abs_grad_x, 0.5, abs_grad_y,0.5, 0, edges);
    // imshow("edges",edges);
    // cv::threshold(edges, binaryImage, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
    // imshow("binaryImage",binaryImage);
    //--------------------------------------------------赛道条件差的情况----------------------------------------------
    // Sobel边缘检测
    Mat originalImage;
    cvtColor(frame, originalImage, cv::COLOR_BGR2GRAY);
    // float alpha = 0.2;  // 调整这个值以控制对比度增强的强度
    // Mat enhancedImage = customEqualizeHist(originalImage, alpha);
    Mat enhancedImage = originalImage;
    // Sobel边缘检测
    Mat sobelx, sobely;
    Sobel(enhancedImage, sobelx, CV_64F, 1, 0, 3);
    Sobel(enhancedImage, sobely, CV_64F, 0, 1, 3);

    Mat gradientMagnitude = abs(sobelx) + abs(sobely);
    convertScaleAbs(gradientMagnitude, gradientMagnitude);

    // 调整阈值
    Mat binaryImage12 = Mat::zeros(enhancedImage.size(), CV_8U);
    // threshold(gradientMagnitude, binaryImage12, 50, 255, THRESH_BINARY);
    cv::threshold(gradientMagnitude, binaryImage12, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
    //-------------------------------------------------上新方案----------------------------------------------
    cv::Mat kernel1 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::dilate(binaryImage12, binaryImage, kernel1, cv::Point(-1, -1), 1); // 这个地方也要修改
    // cv::dilate(binaryImage, binaryImage, kernel1, cv::Point(-1, -1), 1);
    int x_roi = 1;
    int y_roi = 109;
    int width_roi = 318;
    int height_roi = 46;
    cv::Rect roi(x_roi, y_roi, width_roi, height_roi);

    cv::Mat croppedObject = binaryImage(roi);
    vector<Vec4i> lines;
    HoughLinesP(croppedObject, lines, 1, CV_PI / 180, 25, 15, 10);
    for (size_t i = 0; i < lines.size(); i++)
    {
        Vec4i l = lines[i];
        double angle = atan2(l[3] - l[1], l[2] - l[0]) * 180.0 / CV_PI;
        double length = sqrt(pow(l[3] - l[1], 2) + pow(l[2] - l[0], 2));
        double aspect_ratio = length / abs(l[3] - l[1]);
        if (abs(angle) > 15)
        {
            Vec4i l = lines[i];
            l[0] += x_roi;
            l[1] += y_roi;
            l[2] += x_roi;
            l[3] += y_roi;
            line(binaryImage_1, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(255), 2, LINE_AA);
        }
    }
    return binaryImage_1;
}
// --------------------------------------------------------新方案---------------------------------------------------

void Tracking(const cv::Mat &dilated_image)
{
    int begin = 160;
    left_line.clear();
    right_line.clear();
    mid.clear();
    for (int i = 153; i >= 110; i--)
    {
        int find_l = 0;
        int find_r = 0;
        int to_left = begin;
        int to_right = begin;

        while (to_left != 1)
        {
            if (dilated_image.at<uchar>(i, to_left) == 255 && dilated_image.at<uchar>(i, to_left + 1) == 255)
            {
                find_l = 1;
                left_line.push_back(cv::Point(to_left, i));
                break;
            }
            else
            {
                to_left--;
            }
        }

        if (to_left == 1)
        {
            left_line.push_back(cv::Point(1, i));
        }

        while (to_right != 318)
        {
            if (dilated_image.at<uchar>(i, to_right) == 255 && dilated_image.at<uchar>(i, to_right - 2) == 255)
            {
                find_r = 1;
                right_line.push_back(cv::Point(to_right, i));
                break;
            }
            else
            {
                to_right++;
            }
        }

        if (to_right == 318)
        {
            right_line.push_back(cv::Point(318, i));
        }
        cv::Point midx1 = left_line.back();
        cv::Point midx2 = right_line.back();
        mid.push_back(cv::Point(int((midx1.x + midx2.x) / 2), i));
        begin = (to_right + to_left) / 2;
    }
}
void Tracking_blue(const cv::Mat &dilated_image)
{
    int begin = 160;
    left_line_blue.clear();
    right_line_blue.clear();
    mid_blue.clear();
    for (int i = 153; i >= heighest; i--)
    {
        int find_l = 0;
        int find_r = 0;
        int to_left = begin;
        int to_right = begin;

        while (to_left != 1)
        {
            if (dilated_image.at<uchar>(i, to_left) == 255 && dilated_image.at<uchar>(i, to_left + 1) == 255)
            {
                find_l = 1;
                left_line_blue.push_back(cv::Point(to_left, i));
                break;
            }
            else
            {
                to_left--;
            }
        }

        if (to_left == 1)
        {
            left_line_blue.push_back(cv::Point(1, i));
        }

        while (to_right != 318)
        {
            if (dilated_image.at<uchar>(i, to_right) == 255 && dilated_image.at<uchar>(i, to_right - 2) == 255)
            {
                find_r = 1;
                right_line_blue.push_back(cv::Point(to_right, i));
                break;
            }
            else
            {
                to_right++;
            }
        }

        if (to_right == 318)
        {
            right_line_blue.push_back(cv::Point(318, i));
        }
        cv::Point midx1 = left_line_blue.back();
        cv::Point midx2 = right_line_blue.back();
        mid_blue.push_back(cv::Point(int((midx1.x + midx2.x) / 2), i));
        begin = (to_right + to_left) / 2;
    }
}
void Tracking_blue_zhuitong1(const cv::Mat &dilated_image)
{
    int begin = 210;
    left_line.clear();
    right_line.clear();
    mid.clear();
    for (int i = 153; i >= heighest; i--)
    {
        int find_l = 0;
        int find_r = 0;
        int to_left = begin;
        int to_right = begin;

        while (to_left != 1)
        {
            if (dilated_image.at<uchar>(i, to_left) == 255 && dilated_image.at<uchar>(i, to_left + 1) == 255)
            {
                find_l = 1;
                left_line.push_back(cv::Point(to_left, i));
                break;
            }
            else
            {
                to_left--;
            }
        }

        if (to_left == 1)
        {
            left_line.push_back(cv::Point(1, i));
        }

        while (to_right != 318)
        {
            if (dilated_image.at<uchar>(i, to_right) == 255 && dilated_image.at<uchar>(i, to_right - 2) == 255)
            {
                find_r = 1;
                right_line.push_back(cv::Point(to_right, i));
                break;
            }
            else
            {
                to_right++;
            }
        }

        if (to_right == 318)
        {
            right_line.push_back(cv::Point(318, i));
        }
        cv::Point midx1 = left_line.back();
        cv::Point midx2 = right_line.back();
        mid.push_back(cv::Point(int((midx1.x + midx2.x) / 2), i));
        begin = (to_right + to_left) / 2;
    }
}
void Tracking_blue_zhuitong2(const cv::Mat &dilated_image)
{
    int begin = 120;
    left_line.clear();
    right_line.clear();
    mid.clear();
    for (int i = 153; i >= heighest; i--)
    {
        int find_l = 0;
        int find_r = 0;
        int to_left = begin;
        int to_right = begin;

        while (to_left != 1)
        {
            if (dilated_image.at<uchar>(i, to_left) == 255 && dilated_image.at<uchar>(i, to_left + 1) == 255)
            {
                find_l = 1;
                left_line.push_back(cv::Point(to_left, i));
                break;
            }
            else
            {
                to_left--;
            }
        }

        if (to_left == 1)
        {
            left_line.push_back(cv::Point(1, i));
        }

        while (to_right != 318)
        {
            if (dilated_image.at<uchar>(i, to_right) == 255 && dilated_image.at<uchar>(i, to_right - 2) == 255)
            {
                find_r = 1;
                right_line.push_back(cv::Point(to_right, i));
                break;
            }
            else
            {
                to_right++;
            }
        }

        if (to_right == 318)
        {
            right_line.push_back(cv::Point(318, i));
        }
        cv::Point midx1 = left_line.back();
        cv::Point midx2 = right_line.back();
        mid.push_back(cv::Point(int((midx1.x + midx2.x) / 2), i));
        begin = (to_right + to_left) / 2;
    }
}
float servo_pd_blue(int target)
{
    int pidx = mid_blue[(int)(mid_blue.size() / 2)].x;
    cout << "servo_pd_blue" << pidx << endl;
    error_first = target - pidx;
    servo_pwm_diff = 3.0 * error_first + 3.0 * (error_first - last_error);
    last_error = error_first;
    if (servo_pwm_diff < 0)
    {
        servo_pwm_diff = servo_pwm_diff;
    }
    else
    {
        servo_pwm_diff = servo_pwm_diff;
    }
    servo_pwm = 720 + servo_pwm_diff;
    if (servo_pwm > 1000)
    {
        servo_pwm = 1000;
    }
    else if (servo_pwm < 500)
    {
        servo_pwm = 500;
    }
    return servo_pwm;
}
float servo_pd_after(int target)
{
    int size = int(mid.size());

    // cout << "size:" << size << endl;
    int pidx = int((mid[23].x + mid[20].x + mid[25].x) / 3);
    error_first = target - pidx;
    servo_pwm_diff = kp * error_first + kd * (error_first - last_error);
    // cout << "servo_pwm_diff:" << servo_pwm_diff << endl;
    last_error = error_first;
    servo_pwm = 710 + servo_pwm_diff;
    if (servo_pwm > 900)
    {
        servo_pwm = 900;
    }
    else if (servo_pwm < 600)
    {
        servo_pwm = 600;
    }
    return servo_pwm;
}
float servo_pd(int target)
{
    int size = int(mid.size());
    int pidx = int((mid[23].x + mid[20].x + mid[25].x) / 3);
    error_first = target - pidx;
    servo_pwm_diff = kp * error_first + kd * (error_first - last_error);
    // cout << "servo_pwm_diff:" << servo_pwm_diff << endl;
    last_error = error_first;
    servo_pwm = 720 + servo_pwm_diff;
    if (servo_pwm > 900)
    {
        servo_pwm = 900;
    }
    else if (servo_pwm < 600)
    {
        servo_pwm = 600;
    }
    return servo_pwm;
}
float servo_pd_fast(int target)
{
    int size = int(mid.size());
    // cout << "size:" << size << endl;
    int pidx = int((mid[23].x + mid[20].x + mid[25].x) / 3);
    // int pidx = int((mid[line_mid].x + mid[line_mid - 5].x + mid[line_mid - 10].x) / 3);
    error_first = target - pidx;
    servo_pwm_diff = 1.0 * error_first + 2.0 * (error_first - last_error);
    // cout << "servo_pwm_diff:" << servo_pwm_diff << endl;
    last_error = error_first;
    servo_pwm = 720 + servo_pwm_diff;
    if (servo_pwm > 1000)
    {
        servo_pwm = 1000;
    }
    else if (servo_pwm < 500)
    {
        servo_pwm = 500;
    }
    return servo_pwm;
}
double average(vector<int> vec)
{
    if (vec.size() < 1)
        return -1;

    double sum = 0;
    for (int i = 0; i < vec.size(); i++)
    {
        sum += vec[i];
    }

    return (double)sum / vec.size();
}
double sigma(vector<int> vec)
{
    if (vec.size() < 1)
        return 0;

    double aver = average(vec);
    double sigma = 0;
    for (int i = 0; i < vec.size(); i++)
    {
        sigma += (vec[i] - aver) * (vec[i] - aver);
    }
    sigma /= (double)vec.size();
    return sigma;
}

void motor_servo_contral(int flag_yellow_cond, int cross)
{
    //-----------------------------------------锥桶方案-----------------------------------------------------
    if (flag_yellow_cond != 0 && flag_yellow_finish == 0 && flag_cross_finish == 1)
    {
        flag_yellow_finish = 1;
        gpioPWM(12, 730);
        gpioPWM(13, 10000);
        usleep(250000);
        gpioPWM(13, 9800);
        gpioPWM(13, 8800);
        usleep(250000);
        _exit(0);
    }
    else if (cross == 1 && flag_cross_finish == 0)
    {
        flag_cross_finish = 1;
        gpioPWM(13, 9800);
        gpioPWM(13, 8900);
        usleep(550000);
        gpioPWM(12, 730);
        gpioPWM(13, 10000);
        sleep(3);
    }
    else if (contours_all.size() != 0 && count_change < 3 && number1 >= 7 && flag_cross_finish == 1)
    {
        if (try_patching_line == 2 && count_change < 3 && count_change >= 1)
        {
            float servo_pwm_chayan = servo_pd_blue(160);
            servo_pwm_now = servo_pwm_chayan;
        }
        else if (try_patching_line == 1 && count_change >= 1 && count_change < 3)
        {
            float servo_pwm_chayan = servo_pd_blue(160);
            servo_pwm_now = servo_pwm_chayan;
        }
        else
        {
            servo_pwm_now = servo_pd_blue(160);
        }
        cout << "bin_imagepwm" << servo_pwm_now << endl;
        cout << "------------------------" << endl;
        if (times4 == 0)
        {
            times4 = 1;
            // gpioPWM(13, 9800);
            // gpioPWM(13, 8700);
            gpioPWM(13, 10000);
            // usleep(50000);
        }
        // gpioPWM(13, 10000);//11000
        // gpioPWM(13, 10500);
        gpioPWM(12, servo_pwm_now);
    }
    else
    {
        if (count_change < 1 || count_change > 2)
        {
            cout << 1 << endl;
            if (count_change > 2 && flag_cross_finish == 1)
            {
                servo_pwm_now = servo_pd(160);
                cout << "pwm" << servo_pwm_now << endl;
                gpioPWM(13, 13000);
                gpioPWM(12, servo_pwm_now);
            }
            else if (count_change < 1 && flag_cross_finish == 0)
            {
                servo_pwm_now = servo_pd(160);
                cout << "pwm" << servo_pwm_now << endl;
                gpioPWM(13, 12000);
                gpioPWM(12, servo_pwm_now);
            }
            else
            {
                cout << "after" << endl;
                servo_pwm_now = servo_pd_after(160);
                cout << "pwm" << servo_pwm_now << endl;
                gpioPWM(13, 11700);
                gpioPWM(12, servo_pwm_now);
            }
        }
        else
        {
            cout << 2 << endl;
            servo_pwm_now = servo_pd_left(160);
            cout << "pwm" << servo_pwm_now << endl;
            gpioPWM(13, 11400);
            gpioPWM(12, servo_pwm_now);
        }
    }
    // -----------------------------------------锥桶方案-----------------------------------------------------

    // ------------------------------------纯竞速---------------------------------
    // if (flag_yellow_cond != 0 && flag_yellow_finish == 0 && flag_cross_finish == 1)
    // {
    //     flag_yellow_finish = 1;
    //     gpioPWM(12, 730);
    //     gpioPWM(13, 10000);
    //     usleep(250000);
    //     gpioPWM(13, 9800);
    //     gpioPWM(13, 8800);
    //     usleep(250000);
    //     cout << "停止" << endl;
    //     _exit(0);
    // }
    // else if (cross == 1 && flag_cross_finish == 0)
    // {
    //     flag_cross_finish = 1;
    //     cout << "11" << endl;
    //     gpioPWM(13, 9800);
    //     gpioPWM(13, 8900);
    //     usleep(550000); //
    //     gpioPWM(12, 730);
    //     gpioPWM(13, 10000);
    //     sleep(3);
    // }
    // else if (contours_all.size() != 0 && count_change < 3 && number1 >= 7)
    // {
    //     if (try_patching_line == 2 && count_change < 3 && count_change >= 1)
    //     {
    //         float servo_pwm_chayan = servo_pd_blue(160);
    //         servo_pwm_now = servo_pwm_chayan;
    //     }
    //     else if (try_patching_line == 1 && count_change >= 1 && count_change < 3)
    //     {
    //         float servo_pwm_chayan = servo_pd_blue(160);
    //         servo_pwm_now = servo_pwm_chayan;
    //     }
    //     else
    //     {
    //         servo_pwm_now = servo_pd_blue(160);
    //     }
    //     if (times4 == 0)
    //     {
    //         times4 = 1;
    //         gpioPWM(13, 9800);
    //         gpioPWM(13, 8700);
    //         usleep(100000);
    //     }
    //     gpioPWM(13, 10000);//11000
    //     gpioPWM(12, servo_pwm_now);
    // }
    // else
    // {
    //     if (flag_cross_finish == 0)
    //     {
    //         servo_pwm_now = servo_pd(160);
    //         gpioPWM(13, 12000);
    //         gpioPWM(12, servo_pwm_now);

    //     }
    //     else
    //     {
    //         servo_pwm_now = servo_pd(160);
    //         gpioPWM(13, 12500);
    //         gpioPWM(12, servo_pwm_now);

    //     }
    // }
}
void draw_point()
{
    for (const cv::Point &point : left_line)
    {
        cv::circle(frame, point, 5, cv::Scalar(0, 0, 255), -1);
    }
    for (const cv::Point &point : right_line)
    {
        cv::circle(frame, point, 5, cv::Scalar(0, 255, 0), -1);
    }
    for (const cv::Point &point : mid)
    {
        cv::circle(frame, point, 5, cv::Scalar(255, 0, 0), -1);
    }
}
void draw_point_blue()
{
    for (const cv::Point &point : left_line_blue)
    {
        cv::circle(frame, point, 5, cv::Scalar(0, 0, 255), -1);
    }
    for (const cv::Point &point : right_line_blue)
    {
        cv::circle(frame, point, 5, cv::Scalar(0, 255, 0), -1);
    }
    for (const cv::Point &point : mid_blue)
    {
        cv::circle(frame, point, 5, cv::Scalar(255, 0, 0), -1);
    }
}

//------------------------------------------------------------------------------------------------------------------------------------

void computer_area(vector<vector<Point>> contours_all)
{
    Point2f center;
    float radius;
    for (int i = 0; i < contours_all.size(); i++)
    {
        minEnclosingCircle(contours_all[i], center, radius);
        circle(frame, center, 8, Scalar(0, 0, 255), -1);
    }
}

bool Contour_Area(vector<Point> contour1, vector<Point> contour2)
{
    return contourArea(contour1) > contourArea(contour2);
}

vector<Point> blue_vertebral_barrel_find(Mat mask)
{
    vector<vector<Point>> contours;
    vector<Vec4i> hierarcy;
    findContours(mask, contours, hierarcy, RETR_EXTERNAL, CHAIN_APPROX_NONE);
    sort(contours.begin(), contours.end(), Contour_Area);
    return contours[0];
}
vector<vector<Point>> blue_vertebral_barrel_find_all(Mat mask)
{
    vector<vector<Point>> contours;
    vector<Vec4i> hierarcy;
    Point2f center;
    float radius;
    findContours(mask, contours, hierarcy, RETR_EXTERNAL, CHAIN_APPROX_NONE);
    if (contours.size() > 0)
    {
        vector<vector<Point>> newContours;
        for (const vector<Point> &contour : contours)
        {
            Point2f center;
            float radius;
            minEnclosingCircle(contour, center, radius);
            if (center.y > 108 && center.y < 153)
            {
                newContours.push_back(contour);
            }
        }
        contours = newContours;
    }
    if (contours.size() > 0)
    {
        vector<vector<Point>> newContours2;
        for (const vector<Point> &contour : contours)
        {
            if (contourArea(contour) > 10)
            {
                newContours2.push_back(contour);
            }
        }
        contours = newContours2;
    }
    if (contours.size() > 0)
    {
        vector<vector<Point>> newContours5;
        for (const vector<Point> &contour : contours)
        {
            Point2f center;
            float radius;
            minEnclosingCircle(contour, center, radius);
            center.x = (int)center.x;
            center.y = (int)center.y;
            if (center.x > left_line[center.y - 108].x && center.x < right_line[center.y - 108].x)
            {
                cout << "过滤后的点：center.x" << center.x << endl;
                cout << "center.y " << center.y << endl;
                cout << "left_line " << left_line[center.y - 108].x << endl;
                cout << "right_line " << right_line[center.y - 108].x << endl;
                cv::circle(frame, Point(center.x, center.y), 10, cv::Scalar(0, 0, 0), -1);
                newContours5.push_back(contour);
            }
        }
        contours = newContours5;
    }
    if (contours.size() > 0)
    {
        sort(contours.begin(), contours.end(), Contour_Area);
        now_blue_max = (int)contourArea(contours[0]);
    }
    else
    {
        now_blue_max = 0;
    }
    vector<vector<Point>> newContours4;
    newContours4 = contours;
    if (contours.size() > 0)
    {
        vector<vector<Point>> newContours3;
        for (const vector<Point> &contour : contours)
        {
            if (contourArea(contour) < 140)
            {
                newContours3.push_back(contour);
            }
        }
        contours = newContours3;
    }
    // cout << "now_blue_max" << now_blue_max <<endl;
    // cout << "contours.size()" << contours.size() <<endl;
    if (contours.size() == 0 && newContours4.size() != 0)
    {
        if (last_blue == 0)
        {
            if (try_patching_line == 1)
            {
                try_patching_line = 2;
            }
            else if (try_patching_line == 2)
            {
                try_patching_line = 1;
            }
            cout << "--------------------------------补线方式转换------------------------------" << endl;
            number1 = 0;
            count_change++;
        }
    }
    if (now_blue_max > 140)
    {
        last_blue = 1;
    }
    else
    {
        last_blue = 0;
    }
    return contours;
}

Point pic_frame(Mat &frame, Point center, int radius)
{
    // circle(frame, center, radius, Scalar(0, 255, 0), 2);
    // circle(frame, center,8, Scalar(0, 0, 255), -1);
    //    cout << "x : "<<int(center.x) << endl;
    //    cout << "y : "<<int(center.y) << endl;
    return Point(int(center.x), int(center.y));
}

void blue_vertebral_model(void)
{
    contours_all = blue_vertebral_barrel_find_all(mask);
    if (contours_all.size() != 0)
    {
        Point2f center;
        float radius;
        minEnclosingCircle(contours_all[0], center, radius);
        heighest = center.y;
        if (try_patching_line == 2)
        {
            bin_image2 = drawWhiteLine(bin_image2, Point(int(center.x), int(center.y)), Point(int((right_line[0].x + right_line[1].x + right_line[2].x) / 3), 155),
                                       8);
            cout << "center.x:" << center.x << endl;
            cout << "center.y:" << center.y << endl;
            cout << "1" << endl;
        }
        else if (try_patching_line == 1)
        {
            if (count_change != 2)
            {
                bin_image2 = drawWhiteLine(bin_image2, Point(int(center.x), int(center.y)), Point(int((left_line[0].x + left_line[1].x + left_line[2].x) / 3), 155),
                                           8);
            }
            else
            {
                bin_image2 = drawWhiteLine(bin_image2, Point(int(center.x - 20), int(center.y)), Point(int((left_line[0].x + left_line[1].x + left_line[2].x) / 3), 155),
                                           8);
            }
            cout << "center.x:" << center.x << endl;
            cout << "center.y:" << center.y << endl;
            cout << "2" << endl;
        }
    }
}

int crossroad(Mat frame)
{
    flag_cross = 0;
    int height = frame.rows;
    int width = frame.cols;
    Mat hsv;
    cvtColor(frame, hsv, COLOR_BGR2HSV);

    Scalar lower_white = Scalar(0, 0, 221);
    Scalar upper_white = Scalar(180, 30, 255);

    Mat mask1;
    inRange(hsv, lower_white, upper_white, mask1);
    Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
    dilate(mask1, mask1, kernel);
    erode(mask1, mask1, kernel);
    Mat src(mask1, Rect(100, 85, 120, 60));
    int cout1 = 0, cout2 = 0, flag = 0;
    for (int i = 0; i < src.rows; i++)
    {
        if (cout1 < 10)
        {
            flag = 0;
        }
        cout1 = 0;
        for (int j = 10; j < src.cols - 10; j++)
        {
            if (src.at<char>(i, j - 2) == 0 && src.at<uchar>(i, j) == 0 && src.at<uchar>(i, j - 1) == 0 && src.at<uchar>(i, j + 1) == 255 && src.at<uchar>(i, j + 2) == 255)
            {
                cout1++;
            }
            else if (src.at<uchar>(i, j - 2) == 255 && src.at<uchar>(i, j) == 255 && src.at<uchar>(i, j - 1) == 255 && src.at<uchar>(i, j + 1) == 0 && src.at<uchar>(i, j + 2) == 0)
            {
                cout1++;
            }
            if (cout1 >= 10)
            {
                cout2++;
                flag++;
                if (flag >= 3)
                {
                    cout << "斑马线" << endl;
                    flag_cross = 1;
                }
                break;
            }
        }
    }
    cout << "flag_cross" << flag_cross << endl;
    return flag_cross;
}

int yellow_edge(Mat img, bool visual_flag)
{
    Mat cropped_image, canvas, image;
    int yellow_num = 0;
    int height = img.rows;
    int width = img.cols;
    int half_height = int(height / 2);
    int per_height = int(height / 20);
    image = img.clone();
    cropped_image = image(Rect(0, half_height - per_height, width, per_height * 3));
    if (visual_flag == true)
    {
        canvas = cropped_image.clone();
    }
    cvtColor(cropped_image, cropped_image, COLOR_BGR2GRAY);
    Canny(cropped_image, cropped_image, 50, 150, 3);
    vector<Vec4i> lines;
    HoughLinesP(cropped_image, lines, 1, CV_PI / 180, 150, 125, 10);
    if (lines.size() == 0)
    {
        printf("No yellow edge detected!\n");
        return 0;
    }
    else
    {
        for (size_t i = 0; i < lines.size(); i++)
        {
            Vec4i l = lines[i];
            double angle = atan2(l[3] - l[1], l[2] - l[0]) * 180.0 / CV_PI;
            double length = sqrt(pow(l[3] - l[1], 2) + pow(l[2] - l[0], 2));
            double aspect_ratio = length / abs(l[3] - l[1]);
            if (abs(angle) < 5 && aspect_ratio > 5)
            {
                if (visual_flag == true)
                {
                    line(canvas, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(255, 0, 0), 2, LINE_AA);
                    printf("x: %d, y: %d, x1: %d, y1: %d, angle: %f, length: %f, aspect_ratio: %f\n", l[0], l[1], l[2],
                           l[3], angle, length, aspect_ratio);
                }
                yellow_num += 1;
            }
        }
    }
    if (visual_flag == true)
    {
        imshow("Image_edge", canvas);
        waitKey(0);
    }
    return yellow_num;
}

int yellow_hsv(Mat img, bool visual_flag)
{
    Mat cropped_image, canvas, image;
    Scalar lowerb = Scalar(3, 0, 0);
    Scalar upperb = Scalar(40, 100, 255);
    int yellow_num = 0;
    int height = img.rows;
    int width = img.cols;
    int half_height = int(height / 2);
    int per_height = int(height / 20);
    int area_1 = int(0.002775 * height * width);
    int area_2 = int(0.025 * height * width);
    image = img.clone();
    cropped_image = image(Rect(0, half_height - per_height, width, per_height * 3));
    if (visual_flag == true)
    {
        canvas = cropped_image.clone();
    }
    cvtColor(cropped_image, cropped_image, COLOR_BGR2HSV);
    morphologyEx(cropped_image, cropped_image, MORPH_OPEN, getStructuringElement(MORPH_RECT, Size(3, 3)));
    inRange(cropped_image, lowerb, upperb, cropped_image);
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(cropped_image, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    for (size_t i = 0; i < contours.size(); i++)
    {
        double area = contourArea(contours[i]);
        if (area > area_1 && area < area_2)
        {
            Rect rect = boundingRect(contours[i]);
            double aspect_ratio = rect.width / rect.height;
            if (aspect_ratio > 10)
            {
                if (visual_flag == true)
                {
                    rectangle(canvas, rect, Scalar(255, 0, 0), 2, LINE_AA);
                    printf("x: %d, y: %d, width: %d, height: %d, aspect_ratio: %f\n", rect.x, rect.y, rect.width,
                           rect.height, aspect_ratio);
                }
                yellow_num += 1;
            }
        }
    }
    if (visual_flag == true)
    {
        imshow("Image_hsv", canvas);
        waitKey(0);
    }
    return yellow_num;
}
void blue_card_find(void)
{
    vector<vector<Point>> contours;
    vector<Vec4i> hierarcy;
    findContours(mask, contours, hierarcy, RETR_EXTERNAL, CHAIN_APPROX_NONE);
    if (contours.size() > 0)
    {
        sort(contours.begin(), contours.end(), Contour_Area);
        vector<vector<Point>> newContours;
        for (const vector<Point> &contour : contours)
        {
            Point2f center;
            float radius;
            minEnclosingCircle(contour, center, radius);
            if (center.y > 90 && center.y < 160)
            {
                newContours.push_back(contour);
            }
        }

        contours = newContours;

        if (contours.size() > 0)
        {
            if (contourArea(contours[0]) > 500)
            {
                cout << "find biggest blue" << endl;
                Point2f center;
                float radius;
                minEnclosingCircle(contours[0], center, radius);
                circle(frame, center, static_cast<int>(radius), Scalar(0, 255, 0), 2);
                find_first = 1;
            }
            else
            {
                cout << "not found blue" << endl;
            }
        }
    }
    else
    {
        cout << "not found blue" << endl;
    }
}
void blue_card_remove(void)
{
    cout << "entry move blue process" << endl;
    vector<vector<Point>> contours;
    vector<Vec4i> hierarcy;
    findContours(mask, contours, hierarcy, RETR_EXTERNAL, CHAIN_APPROX_NONE);
    if (contours.size() > 0)
    {
        sort(contours.begin(), contours.end(), Contour_Area);
        vector<vector<Point>> newContours;
        for (const vector<Point> &contour : contours)
        {
            Point2f center;
            float radius;
            minEnclosingCircle(contour, center, radius);
            if (center.y > 90 && center.y < 160)
            {
                newContours.push_back(contour);
            }
        }

        contours = newContours;

        if (contours.size() == 0)
        {
            begin_sign = 0;
            cout << "move" << endl;
            sleep(2);
        }
    }
    else
    {
        begin_sign = 0;
        cout << "蓝色挡板移开" << endl;
        sleep(2);
    }
}
int main(void)
{
    bool visual_flag = false;
    int yellow_num_edge = 0;
    int yellow_num_hsv = 0;
    VideoCapture capture;
    gpioTerminate();
    servo_motor_pwmInit();
    capture.open(0);
    if (!capture.isOpened())
    {
        cout << "Can not open video file!" << endl;
        system("pause");
        return -1;
    }
    std::cout << "FPS: " << capture.get(cv::CAP_PROP_FPS) << std::endl;
    std::cout << "Frame Width: " << capture.get(cv::CAP_PROP_FRAME_WIDTH) << std::endl;
    std::cout << "Frame Height: " << capture.get(cv::CAP_PROP_FRAME_HEIGHT) << std::endl;

    capture.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
    capture.set(cv::CAP_PROP_FPS, 30);
    capture.set(cv::CAP_PROP_FRAME_WIDTH, 320);
    capture.set(cv::CAP_PROP_FRAME_HEIGHT, 240);
    auto start_time = std::chrono::high_resolution_clock::now();
    while (capture.read(frame))
    {
        frame = undistort(frame);
        if (begin_sign == 1)
        {
            Mat frame_a = frame.clone();
            cvtColor(frame_a, change_frame, 40);
            Scalar scalarl = Scalar(100, 43, 46);
            Scalar scalarH = Scalar(124, 255, 255);
            inRange(change_frame, scalarl, scalarH, mask);
            Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
            morphologyEx(mask, mask, MORPH_OPEN, kernel);
            morphologyEx(mask, mask, MORPH_CLOSE, kernel);
            for (int y = 0; y < mask.rows; y++)
            {
                for (int x = 0; x < mask.cols; x++)
                {
                    if (y < 90 || y > 160)
                    {
                        mask.at<uchar>(y, x) = 0;
                    }
                }
            }
            if (find_first == 0)
            {
                blue_card_find();
            }
            else
            {
                blue_card_remove();
            }
        }
        else
        {
            auto end_time = std::chrono::high_resolution_clock::now();
            Mat frame_a = frame.clone();
            cvtColor(frame_a, change_frame, 40);
            Scalar scalarl = Scalar(100, 43, 46);
            Scalar scalarH = Scalar(124, 255, 255);
            inRange(change_frame, scalarl, scalarH, mask);
            Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
            morphologyEx(mask, mask, MORPH_OPEN, kernel);
            morphologyEx(mask, mask, MORPH_CLOSE, kernel);
            cv::cvtColor(frame, grayImage, cv::COLOR_BGR2GRAY);
            bin_image = ImagePreprocessing(frame);
            if (flag_cross_finish == 1)
            {
                yellow_num_hsv = yellow_hsv(frame, visual_flag);
            }
            Tracking(bin_image);
            bin_image2 = bin_image.clone();
            number1++;
            if (number1 > 1000)
            {
                number1 = 100;
            }
            if (flag_cross_finish == 0)
            {
                if (number1 > 10)
                {
                    cross = crossroad(frame);
                }
            }
            if (number1 > 10 && flag_cross_finish == 1)
            {
                blue_vertebral_model();
            }
            if (count_change < 3)
            {
                if (contours_all.size() != 0 && number1 >= 7 && flag_cross_finish == 1)
                {
                    cout << heighest << endl;
                    Tracking_blue(bin_image2);
                }
            }
            motor_servo_contral(yellow_num_hsv, cross);
            // cout << "count_change"<< count_change <<endl;
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
            std::cout << " times：" << duration.count() / 1000 << "s" << std::endl;
            start_time = end_time;
            // if(contours_all.size() != 0)
            // {
            //     draw_point_blue();
            // }
            // else
            // {
            // draw_point();
            //  }
            // imshow("frame", frame);
            // imshow("frame2", bin_image);
            // imshow("frame3", bin_image2);
            // int key = cv::waitKey(1);

            // if (key == 27)
            // {
            //    break;
            //  }
        }
    }
    capture.release();
    destroyAllWindows();
    system("pause");
    return 0;
}
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

Mat g_mask1Image, g_midImage, g_mask1GrayImage;
cv::Mat sobelX8U;
cv::Mat blurredImage;
cv::Mat sobelX;
cv::Mat binaryImage;
cv::Mat croppedObject;
int g_HoughLinesThreshold = 150;
int g_minLineLength = 100;

void on_HoughLines(int, void *);
Mat customEqualizeHist(const Mat &inputImage, float alpha)
{
    Mat enhancedImage;
    equalizeHist(inputImage, enhancedImage);

    // 减弱对比度增强的效果
    return alpha * enhancedImage + (1 - alpha) * inputImage;
}
int main()
{
    g_mask1Image = imread("/home/pi/test_photo/1.jpg", cv::IMREAD_GRAYSCALE);
    if (!g_mask1Image.data)
    {
        return -1;
    }

    imshow("g_mask1Image", g_mask1Image);
    // float alpha = 0.5;  // 调整这个值以控制对比度增强的强度
    // Mat enhancedImage = customEqualizeHist(g_mask1Image, alpha);
    Mat enhancedImage = g_mask1Image;
    // Sobel边缘检测
    Mat sobelx, sobely;
    Sobel(enhancedImage, sobelx, CV_64F, 1, 0, 3);
    Sobel(enhancedImage, sobely, CV_64F, 0, 1, 3);

    Mat gradientMagnitude = abs(sobelx) + abs(sobely);
    convertScaleAbs(gradientMagnitude, gradientMagnitude);

    // 调整阈值
    Mat binaryImage12 = Mat::zeros(enhancedImage.size(), CV_8U);
    // threshold(gradientMagnitude, binaryImage12, 50, 255, THRESH_BINARY);
    cv::threshold(gradientMagnitude, binaryImage12, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
    imshow("binaryImage", binaryImage12);
    imshow("gradientMagnitude", gradientMagnitude);
    cv::Mat kernel1 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::dilate(binaryImage12, binaryImage, kernel1, cv::Point(-1, -1), 1);
    // imshow("1",binaryImage);
    int x_roi = 1;
    int y_roi = 109;
    int width_roi = 318;
    int height_roi = 45;
    cv::Rect roi(x_roi, y_roi, width_roi, height_roi);

    croppedObject = binaryImage(roi);
    namedWindow("HoughLines", WINDOW_AUTOSIZE);
    createTrackbar(" g_HoughLinesThreshold", "HoughLines", &g_HoughLinesThreshold, 150, on_HoughLines);
    createTrackbar("g_minLineLength", "HoughLines", &g_minLineLength, 100, on_HoughLines);

    on_HoughLines(0, 0);

    while (char(waitKey(1)) != 'q')
    {
    }

    waitKey(0);
    return 0;
}

void on_HoughLines(int, void *)
{
    cvtColor(croppedObject, g_mask1GrayImage, COLOR_GRAY2BGR);

    vector<Vec4i> lines
        HoughLinesP(croppedObject, lines, 1, CV_PI / 180, g_HoughLinesThreshold, g_minLineLength, 10);

    for (size_t i = 0; i < lines.size(); i++)
    {
        Vec4i l = lines[i];
        line(g_mask1GrayImage, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(186, 88, 255), 1, LINE_AA);
    }

    imshow("HoughLines", g_mask1GrayImage);
}

#include <opencv2/opencv.hpp>

using namespace cv;

// 回调函数
void onTrackbarChange(int, void *)
{
    // 什么也不做
}

int main()
{
    // 读取图片
    Mat img_original = imread("/home/pi/test_photo/1.jpg");

    // 创建窗口
    namedWindow("Canny");

    // 创建两个滑动条，分别控制threshold1和threshold2
    int threshold1 = 50;
    int threshold2 = 100;

    createTrackbar("threshold1", "Canny", &threshold1, 400, onTrackbarChange);
    createTrackbar("threshold2", "Canny", &threshold2, 400, onTrackbarChange);

    while (true)
    {
        // Canny边缘检测
        Mat img_edges;
        Canny(img_original, img_edges, threshold1, threshold2);

        // 显示图片
        imshow("original", img_original);
        imshow("Canny", img_edges);

        // 检测键盘输入，如果按下 'q' 键，退出循环
        if (waitKey(1) == 'q')
        {
            break;
        }
    }

    destroyAllWindows();

    return 0;
}
