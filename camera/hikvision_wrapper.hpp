#ifndef HIKVISION_WRAPPER_HPP
#define HIKVISION_WRAPPER_HPP

#include <opencv2/opencv.hpp>
#include "common_struct.hpp"

class HikVisionWrapper {
public:
    /**
     * @brief 构造函数
     * @param params 摄像头参数配置
     */
    explicit HikVisionWrapper(const s_camera_params& params);

    /**
     * @brief 初始化摄像头
     * @return 是否初始化成功
     */
    bool initialize();

    /**
     * @brief 获取一帧图像
     * @param[out] frame 输出的图像帧
     * @param camera_id 摄像头ID (默认为0)
     * @return 是否获取成功
     */
    bool getFrame(cv::Mat& frame, int camera_id = 0);

    /**
     * @brief 释放摄像头资源
     */
    void release();

    /**
     * @brief 析构函数
     */
    ~HikVisionWrapper();

private:
    class Impl; // 前置声明实现类
    Impl* pImpl; // PIMPL指针
};

#endif // HIKVISION_WRAPPER_HPP
