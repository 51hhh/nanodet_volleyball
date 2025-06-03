#include "hikvision_wrapper.hpp"
#include "MvCameraControl.h"
#include <memory>
#include <iostream>

class HikVisionWrapper::Impl {
public:
    explicit Impl(const s_camera_params& params) : params_(params), handle(nullptr) {}

    bool initialize() {
        MV_CC_DEVICE_INFO_LIST stDeviceList;
        memset(&stDeviceList, 0, sizeof(MV_CC_DEVICE_INFO_LIST));

        // 枚举设备
        nRet = MV_CC_EnumDevices(MV_USB_DEVICE, &stDeviceList);
        if (MV_OK != nRet) {
            std::cerr << "MV_CC_EnumDevices fail! nRet [" << nRet << "]" << std::endl;
            return false;
        }

        if (stDeviceList.nDeviceNum == 0) {
            std::cerr << "Find No Devices!" << std::endl;
            return false;
        }

        // 选择设备并创建句柄
        nRet = MV_CC_CreateHandle(&handle, stDeviceList.pDeviceInfo[params_.device_id]);
        if (MV_OK != nRet) {
            std::cerr << "MV_CC_CreateHandle fail! nRet [" << nRet << "]" << std::endl;
            return false;
        }

        // 打开设备
        nRet = MV_CC_OpenDevice(handle);
        if (MV_OK != nRet) {
            std::cerr << "MV_CC_OpenDevice fail! nRet [" << nRet << "]" << std::endl;
            return false;
        }

        // 设置触发模式为off
        nRet = MV_CC_SetEnumValue(handle, "TriggerMode", 0);
        if (MV_OK != nRet) {
            std::cerr << "MV_CC_SetTriggerMode fail! nRet [" << nRet << "]" << std::endl;
            return false;
        }

        // 开始取流
        nRet = MV_CC_StartGrabbing(handle);
        if (MV_OK != nRet) {
            std::cerr << "MV_CC_StartGrabbing fail! nRet [" << nRet << "]" << std::endl;
            return false;
        }

        // 获取数据包大小
        nRet = MV_CC_GetIntValue(handle, "PayloadSize", &stParam);
        if (MV_OK != nRet) {
            std::cerr << "Get PayloadSize fail! nRet [" << nRet << "]" << std::endl;
            return false;
        }

        // 分配图像缓冲区
        memset(&stImageInfo, 0, sizeof(MV_FRAME_OUT_INFO_EX));
        pData = (unsigned char *)malloc(sizeof(unsigned char) * stParam.nCurValue);
        if (nullptr == pData) {
            std::cerr << "Failed to allocate image buffer" << std::endl;
            return false;
        }
        nDataSize = stParam.nCurValue;

        return true;
    }

    bool getFrame(cv::Mat& frame, int camera_id) {
        nRet = MV_CC_GetOneFrameTimeout(handle, pData, nDataSize, &stImageInfo, 1000);
        if (nRet != MV_OK) {
            return false;
        }

        cv::Mat img_bayerrg = cv::Mat(stImageInfo.nHeight, stImageInfo.nWidth, CV_8UC1, pData);
        if (camera_id == 0) {
            cv::cvtColor(img_bayerrg, frame, cv::COLOR_BayerRG2RGB);
        } else if (camera_id == 1) {
            cv::Mat img_rgb;
            cv::cvtColor(img_bayerrg, img_rgb, cv::COLOR_BayerRG2RGB);
            cv::resize(img_rgb, frame, cv::Size(640, 640));
        }

        return true;
    }

    void release() {
        if (handle) {
            // 停止取流
            MV_CC_StopGrabbing(handle);
            // 关闭设备
            MV_CC_CloseDevice(handle);
            // 销毁句柄
            MV_CC_DestroyHandle(handle);
            handle = nullptr;
        }

        if (pData) {
            free(pData);
            pData = nullptr;
        }
    }

    ~Impl() {
        release();
    }

private:
    s_camera_params params_;
    void* handle;
    int nRet;
    MVCC_INTVALUE stParam;
    MV_FRAME_OUT_INFO_EX stImageInfo;
    unsigned char* pData;
    unsigned int nDataSize;
};

// 包装类实现
HikVisionWrapper::HikVisionWrapper(const s_camera_params& params) 
    : pImpl(new Impl(params)) {}

bool HikVisionWrapper::initialize() {
    return pImpl->initialize();
}

bool HikVisionWrapper::getFrame(cv::Mat& frame, int camera_id) {
    return pImpl->getFrame(frame, camera_id);
}

void HikVisionWrapper::release() {
    pImpl->release();
}

HikVisionWrapper::~HikVisionWrapper() {
    delete pImpl;
}
