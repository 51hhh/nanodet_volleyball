// 
// Nanodet目标检测演示程序
// 功能：支持图像检测、摄像头实时检测、视频文件检测和性能测试
// 作者：lsf
// 创建时间：2023/5/11
//

#include "Nanodet.h"
#include "../camera/hikvision_wrapper.hpp"
#include "../struct/common_struct.hpp"

/**
 * 图像检测演示
 * @param detector 目标检测器指针
 * @param imagepath 图像文件路径
 * @return 成功返回0，失败返回-1
 */
int image_demo(const std::shared_ptr<NanoDet>& detector, const char* imagepath)
{
    cv::Mat image = cv::imread(imagepath);
    if (image.empty())
    {
        fprintf(stderr, "cv::imread %s failed.\n", imagepath);
        return -1;
    }
    std::vector<Box> boxes;
    detector->detect(image, boxes);
    detector->draw(image, boxes);
    cv::imshow("Nanodet", image);
    cv::waitKey(0);// 按任意键继续
    boxes.clear();
    return 0;
}

/**
 * 摄像头实时检测演示
 * @param detector 目标检测器指针
 * @param cam_id 摄像头设备ID
 * @return 始终返回0
 */
int webcam_demo(const std::shared_ptr<NanoDet>& detector, int cam_id)
{
    // 创建一个空的Mat对象用于存储图像
    cv::Mat image;
    // 定义存储检测框的向量
    std::vector<Box> boxes;
    
    // 配置摄像头参数
    s_camera_params params{cam_id, 1920, 1080, 0, 0, 20};
    
    // 创建并初始化海康摄像头
    HikVisionWrapper hik(params);
    if (!hik.initialize()) {
        fprintf(stderr, "Failed to initialize Hikvision camera %d\n", cam_id);
        return -1;
    }

    // 无限循环进行实时检测
    while (true)
    {
        // 从摄像头读取一帧图像
        if (!hik.getFrame(image)) {
            fprintf(stderr, "Hikvision camera %d get frame failed.\n", cam_id);
            break;
        }
        // 使用统一的时间统计工具
        auto timer = detector->startTimer("Detect");
        detector->detect(image, boxes);
        detector->update_trackers(image,boxes);
        // detector->draw(image, boxes);
        printf("%s\n", detector->endTimer(timer).c_str());

        // // 显示检测结果
        // cv::imshow("Nanodet", image);
        // // 等待键盘输入用于控制程序流
        // cv::waitKey(1);

        // 输出每个检测框的中心坐标
        for (const auto& box : boxes) {
            float center_x = (box.x1 + box.x2) / 2;
            float center_y = (box.y1 + box.y2) / 2;
            printf("Box center: (%.2f, %.2f)\n", center_x, center_y);
        }
        
        // 清空检测框向量
        boxes.clear();
    }
    
    return 0;
}

/**
 * 视频文件检测演示
 * @param detector 目标检测器指针
 * @param path 视频文件路径
 * @return 始终返回0
 */
int video_demo(const std::shared_ptr<NanoDet>& detector, const char* path)
{
    // 读取视频文件
    cv::Mat image;
    cv::VideoCapture cap(path);
    std::vector<Box> boxes;
    // 无限循环检测
    while (true)
    {
        // 读取一帧图像
        bool ret = cap.read(image);
        // 检查图像读取是否成功
        if (!ret) {
            fprintf(stderr, "Video %s read failed.\n", path);
            return 0;
        }
        // 使用统一的时间统计工具
        auto timer = detector->startTimer("Detect");
        detector->detect(image, boxes);
        detector->update_trackers(image,boxes);
        detector->draw(image, boxes);
        printf("%s\n", detector->endTimer(timer).c_str());
        // 显示检测结果
        cv::imshow("Nanodet", image);
        // 等待键盘输入用于控制程序流
        static bool pause = false;
        int key = cv::waitKey(1);
        
        // 空格键切换暂停状态
        if (key == 32) { // 空格键ASCII码
            pause = !pause;
        }
        
        // 暂停状态下按任意键移动一帧
        if (pause) {
            while (true) {
                key = cv::waitKey(0);
                if (key == 32) { // 再次按空格继续
                    pause = !pause;
                    break;
                } else if (key == 27) { // ESC退出
                    return 0;
                } else { // 其他键移动一帧
                    break;
                }
            }
        }
        // 清空检测框向量
        boxes.clear();
    }
}

/**
 * 性能测试函数
 * @param detector 目标检测器指针
 */
void benchmark(const std::shared_ptr<NanoDet>& detector)
{
    int loop_num = 1000;
    detector->benchmark(loop_num);
}


/**
 * 主函数
 * @param argc 参数个数
 * @param argv 参数数组
 * @return 成功返回0，失败返回-1
 * 
 * 参数说明：
 * mode 0: 图像模式，path为图像路径
 * mode 1: 摄像头模式，path为摄像头ID
 * mode 2: 视频模式，path为视频路径
 * mode 3: 性能测试模式，path为0
 */
int main(int argc, char** argv)
{
    if (argc != 3)
    {
        fprintf(stderr, "usage: %s [mode] [path]. \n For image demo, mode=0, path=xxx/xxx/*.jpg; \n For webcam mode=1, path is cam id; \n For video, mode=2; \n For benchmark, mode=3 path=0.\n", argv[0]);
        return -1;
    }

    std::cout << "Start init model." << std::endl;
    // nanodet-full-1.5x-320-int8-ppq.xml  nanodet-1.5x-320-int8.xml
    // 使用构造函数统一设置阈值参数
    auto detector = std::make_shared<NanoDet>(
        "./test/nanodet_IR.xml",
        416, 
        416, 
        NanoDet::PRECISION_FP32, 
        0.8f,  // score_threshold
        0.1f   // nms_threshold
    );
    std::cout << "Init model success." << std::endl;

    // 输入模式webcam0,image1,video2,benchmark3
    int mode = atoi(argv[1]);
    switch (mode)
    {
    case 0:{
        const char* images = argv[2];
        image_demo(detector, images);
        break;
        }
    case 1:{
        int cam_id = atoi(argv[2]);
        webcam_demo(detector, cam_id);
        break;
        }
    case 2:{
        const char* path = argv[2];
        video_demo(detector, path);
        break;
        }
    case 3:{
        benchmark(detector);
        break;
        }
    default:{
        fprintf(stderr, "usage: %s [mode] [path]. \n For webcam mode=0, path is cam id; \n For image demo, mode=1, path=xxx/xxx/*.jpg; \n For video, mode=2; \n For benchmark, mode=3 path=0.\n", argv[0]);
        break;
        }
    }
}
