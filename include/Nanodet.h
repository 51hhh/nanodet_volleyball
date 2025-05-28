//
// Nanodet目标检测头文件
// 定义目标检测相关数据结构和NanoDet类
// 作者：lsf
// 创建时间：2023/5/11
//

#ifndef NANODET_OPENVINO_H
#define NANODET_OPENVINO_H


#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <chrono>
#include <memory>
#include <functional>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>


/**
 * 检测框结构体
 * 包含检测框坐标、置信度、类别和追踪ID
 */
struct Box
{
    // 检测框左上角x坐标
    float x1;
    // 检测框左上角y坐标
    float y1;
    // 检测框右下角x坐标
    float x2;
    // 检测框右下角y坐标
    float y2;
    // 检测框的置信度
    float score;
    // 检测框的类别标签
    int label;
    // 追踪ID
    int track_id;
    
    // 默认构造函数
    Box() : track_id(-1) {}
    // 参数构造函数
    Box(float x1, float y1, float x2, float y2, float score, int label):
        x1(x1), y1(y1), x2(x2), y2(y2), score(score), label(label), track_id(-1) {}
};



/**
 * 卡尔曼滤波追踪器类
 */
class KalmanTracker {
public:
    KalmanTracker(const Box& init_box);
    void predict();
    void update(const Box& measure);
    Box get_state() const;
    int id;
    int time_since_update;
    static int next_id;
    
private:
    cv::KalmanFilter kf;
    Box current_state;
    static std::vector<KalmanTracker> trackers;
    static int next_track_id;
    static float calculate_iou(const Box& a, const Box& b);

};


/**
 * 中心点先验信息结构体
 * 用于目标检测中的特征图处理
 */
struct CenterPrior
{
    int x;
    int y;
    int stride;
};

/**
 * NanoDet目标检测类
 * 基于OpenVINO实现的轻量级目标检测器
 */
class NanoDet
{
public:

    static std::vector<KalmanTracker> trackers;
    static int next_track_id;

    float calculate_iou(const Box& a, const Box& b);
    /**
     * 精度枚举类型
     */
    enum Precision {
        PRECISION_INT8,  // int8量化
        PRECISION_FP16,  // 半精度浮点
        PRECISION_FP32   // 单精度浮点
    };

    /**
     * 构造函数
     * @param model_path 模型文件路径
     * @param width 输入图像宽度
     * @param height 输入图像高度
     * @param precision 模型精度，默认INT8
     * @param score_threshold 分数阈值，默认0.4
     * @param nms_threshold NMS阈值，默认0.5
     */
    explicit NanoDet(const std::string &model_path, int width, int height,
                     Precision precision = PRECISION_INT8,
                     float score_threshold = 0.4, float nms_threshold = 0.5);

    ~NanoDet() = default;

    /**
     * 目标检测接口
     * @param image 输入图像
     * @param boxes_res 输出检测结果
     */
    void detect(cv::Mat& image, std::vector<Box>& boxes_res);


    /**
     * 更新追踪器状态
     * @param image 输入图像
     * @param detections 检测结果
     */
    void update_trackers(cv::Mat& image, const std::vector<Box>& detections);

    /**
     * 绘制检测结果
     * @param image 输入/输出图像
     * @param boxes_res 检测结果
     */
    void draw(cv::Mat& image, std::vector<Box>& boxes_res);

    /**
     * 性能测试
     * @param loop_num 测试循环次数，默认1000
     */
    void benchmark(int loop_num = 1000);

public:
    // 性能统计相关
    struct TimeProfile {
        std::chrono::time_point<std::chrono::steady_clock> start;
        std::string profileName;
    };
    
    /**
     * 开始计时
     * @param name 统计名称
     * @return TimeProfile对象
     */
    TimeProfile startTimer(const std::string& name = "") const {
        return {std::chrono::steady_clock::now(), name};
    }
    
    /**
     * 结束计时并返回耗时(毫秒)
     * @param profile 计时器对象
     * @return 格式化后的时间字符串
     */
    std::string endTimer(const TimeProfile& profile) const {
        auto end = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double, std::milli>(end - profile.start).count();
        if (!profile.profileName.empty()) {
            return profile.profileName + " time: " + std::to_string(elapsed) + " ms";
        }
        return "Time: " + std::to_string(elapsed) + " ms";
    }

private:
    // 轨迹历史记录，key为track_id，value为轨迹点队列(最多200帧)
    std::unordered_map<int, std::deque<cv::Point>> tracking_history_;
    const int TRACK_HISTORY_LENGTH = 200; // 轨迹历史长度

    Precision precision_ = PRECISION_INT8; // 模型精度
    // 模型参数配置
    int input_width_ = 320;  // 模型输入宽度
    int input_height_ = 320; // 模型输入高度
    float score_threshold_ = 0.4; // 检测分数阈值
    float nms_threshold_ = 0.5;  // NMS阈值
    
    // 检测参数
    static constexpr int num_class_ = 1; // 类别数量
    static constexpr int reg_max_ = 7; // 回归最大值
    std::vector<int> strides_ = {8, 16, 32, 64}; // 多尺度特征图步长

    void preprocess(cv::Mat& image);

    void infer();

    void decode_infer();

    void NMS(std::vector<Box>& boxes_res);

    void generate_grid_center_priors();

    ov::InferRequest infer_request_;
    ov::Tensor output_tensor_;
    cv::Mat input_image_;
    float i2d_[6]{}, d2i_[6]{};
    float* output_ptr_{};
    std::vector<Box> Boxes_;
    std::vector<CenterPrior> center_priors_;

    Box disPred2Bbox(float* dfl_det, int label, float score, int x, int y, int stride) const;

    // COCO数据集80类标签
    std::vector<std::string> class_labels_ {
       "volleyball"};

};


#endif //NANODET_OPENVINO_H
