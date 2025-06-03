// 通用结构体定义头文件
// 防止重复包含的宏定义
#ifndef COMMON_STRUCT_HPP
#define COMMON_STRUCT_HPP

// 标准库包含
#include <string>    // 字符串处理
#include <iostream>  // 输入输出流
// OpenCV库包含
#include <opencv2/opencv.hpp>  // OpenCV核心功能

// 调试显示标志，0表示不显示调试信息
#define show 0


// 仿射变换矩阵结构体
// 用于存储2D仿射变换参数
// [v0 v1 v2]
// [v3 v4 v5]
struct AffineMat
{
    float v0, v1, v2;  
    float v3, v4, v5;  
};

// 目标检测结果结构体
// 内存对齐为float大小
struct alignas(float) Detection
{
    float bbox[4];    // 边界框坐标 [x1,y1,x2,y2]
    float conf;       // 检测置信度
    float class_id;   // 类别ID
};

// 相机参数结构体
// 用于存储相机配置参数
typedef struct
{
    int device_id;  // 设备ID
    int width;      // 图像宽度
    int height;     // 图像高度
    int offset_x;   // X轴偏移
    int offset_y;   // Y轴偏移
    int exposure;   // 曝光时间

} s_camera_params;

// 排球检测结果结构体
// 用于存储排球检测信息
typedef struct
{
    float center_x;  // 球心X坐标
    float center_y;  // 球心Y坐标
    float radius;    // 半径
    float deepth;    // 深度信息
    int isValid;     // 是否有效标志 (1有效/0无效)

}volleyball;

// 点集合并存储结构体
// 用于存储合并后的点集信息
typedef struct
{
    int id;                          // 合并ID
    std::vector<cv::Point2f> merge_pts;    // 合并后的点集
    std::vector<float> merge_confs;  // 合并后的置信度
} pick_merge_store;

// 已注释掉的装甲板比较结构体
// 用于按置信度排序装甲板检测结果
// struct armor_compare{
//     bool operator ()(const s_ball& a,const s_ball& b) {
//         return a.conf > b.conf;
//     }
// };

#endif // COMMON_STRUCT_HPP
