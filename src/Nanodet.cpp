/******************************************************************************
 * @file Nanodet.cpp
 * @brief 基于OpenVINO的轻量级目标检测与追踪实现
 * @version 1.0
 * @author lsf
 * @date 2023-05-11
 * 
 * 功能概述：
 * 1. 使用Nanodet模型进行目标检测
 * 2. 基于卡尔曼滤波的多目标追踪
 * 3. 检测结果可视化输出
 *****************************************************************************/

#include "Nanodet.h"

// ==============================================
// KalmanTracker类实现 - 目标状态追踪
// ==============================================

/**
 * @brief 卡尔曼追踪器构造函数
 * @param init_box 初始检测框
 * 
 * 初始化卡尔曼滤波器参数：
 * - 状态维度：6 (x,y,w,h,dx,dy)
 * - 观测维度：4 (x,y,w,h)
 * - 设置状态转移矩阵和观测矩阵
 */
KalmanTracker::KalmanTracker(const Box& init_box) : id(0), time_since_update(0) {
    // 初始化6状态卡尔曼滤波器(中心x,y,宽度,高度,速度x,速度y)
    kf = cv::KalmanFilter(6, 4, 0);
    
    /* 状态转移矩阵A [6x6]
     * 描述状态如何随时间变化:
     * x' = x + dx
     * y' = y + dy
     * w' = w
     * h' = h 
     * dx' = dx
     * dy' = dy
     */
    kf.transitionMatrix = (cv::Mat_<float>(6, 6) << 
        1,0,0,0,1,0,  // x = x + dx
        0,1,0,0,0,1,  // y = y + dy
        0,0,1,0,0,0,  // w = w
        0,0,0,1,0,0,  // h = h
        0,0,0,0,1,0,  // dx = dx
        0,0,0,0,0,1); // dy = dy
    
    // 观测矩阵H [4x6] - 只能观测到位置和大小，不能直接观测速度
    kf.measurementMatrix = (cv::Mat_<float>(4, 6) << 
        1,0,0,0,0,0,  // 观测x
        0,1,0,0,0,0,  // 观测y
        0,0,1,0,0,0,  // 观测w
        0,0,0,1,0,0); // 观测h
    
    // 初始化状态向量 [cx, cy, w, h, dx, dy]
    kf.statePost.at<float>(0) = (init_box.x1 + init_box.x2) / 2; // 中心x
    kf.statePost.at<float>(1) = (init_box.y1 + init_box.y2) / 2; // 中心y
    kf.statePost.at<float>(2) = init_box.x2 - init_box.x1;       // 宽度
    kf.statePost.at<float>(3) = init_box.y2 - init_box.y1;       // 高度
    kf.statePost.at<float>(4) = 0; // 初始x速度
    kf.statePost.at<float>(5) = 0; // 初始y速度
    
    // 设置噪声协方差矩阵
    setIdentity(kf.processNoiseCov, cv::Scalar::all(1e-2));  // 过程噪声
    setIdentity(kf.measurementNoiseCov, cv::Scalar::all(1e-1)); // 观测噪声
    setIdentity(kf.errorCovPost, cv::Scalar::all(1));       // 误差协方差
}

void KalmanTracker::predict() {
    kf.predict();
    time_since_update++;
}

void KalmanTracker::update(const Box& measure) {
    cv::Mat measurement = (cv::Mat_<float>(4,1) << 
        (measure.x1 + measure.x2)/2, // cx
        (measure.y1 + measure.y2)/2, // cy
        measure.x2 - measure.x1,     // w
        measure.y2 - measure.y1);     // h
    kf.correct(measurement);
    time_since_update = 0;
}

Box KalmanTracker::get_state() const {
    float cx = kf.statePost.at<float>(0);
    float cy = kf.statePost.at<float>(1);
    float w = kf.statePost.at<float>(2);
    float h = kf.statePost.at<float>(3);
    
    Box b;
    b.x1 = cx - w/2;
    b.y1 = cy - h/2;
    b.x2 = cx + w/2;
    b.y2 = cy + h/2;
    b.track_id = id;
    return b;
}

// 静态变量初始化
int KalmanTracker::next_id = 0;

float fast_exp(float x)
{
    union {
        uint32_t i;
        float f;
    } v{};
    v.i = (1 << 23) * (1.4426950409 * x + 126.93490512f);
    return v.f;
}


void activation_function_softmax(const float *src, float *dst, int length)
{
    const float alpha = *std::max_element(src, src + length);
    float denominator{0};
    for (int i = 0; i < length; ++i)
    {
        dst[i] = fast_exp(src[i] - alpha);
        denominator += dst[i];
    }
    for (int i = 0; i < length; ++i)
    {
        dst[i] /= denominator;
    }
}


static std::tuple<uint8_t, uint8_t, uint8_t> hsv2bgr(float h, float s, float v){
    const int h_i = static_cast<int>(h * 6);
    const float f = h * 6 - h_i;
    const float p = v * (1 - s);
    const float q = v * (1 - f*s);
    const float t = v * (1 - (1 - f) * s);
    float r, g, b;
    switch (h_i) {
        case 0:r = v; g = t; b = p;break;
        case 1:r = q; g = v; b = p;break;
        case 2:r = p; g = v; b = t;break;
        case 3:r = p; g = q; b = v;break;
        case 4:r = t; g = p; b = v;break;
        case 5:r = v; g = p; b = q;break;
        default:r = 1; g = 1; b = 1;break;}
    return std::make_tuple(static_cast<uint8_t>(b * 255), static_cast<uint8_t>(g * 255), static_cast<uint8_t>(r * 255));
}


static std::tuple<uint8_t, uint8_t, uint8_t> random_color(int id){
    float h_plane = ((((unsigned int)id << 2) ^ 0x937151) % 100) / 100.0f;;
    float s_plane = ((((unsigned int)id << 3) ^ 0x315793) % 100) / 100.0f;
    return hsv2bgr(h_plane, s_plane, 1);
}


NanoDet::NanoDet(const std::string &model_path, int width, int height,
                 Precision precision, float score_threshold, float nms_threshold)
    : precision_(precision)
{
    // 1.创建OpenVINO Runtime Core对象
    ov::Core core;
    // 2.载入并编译模型
    ov::CompiledModel compile_model = core.compile_model(model_path, "CPU");
    // 3.创建推理请求
    infer_request_ = compile_model.create_infer_request();

    // 4. 初始化一些变量
    input_width_ = width;
    input_height_ = height;
    score_threshold_ = score_threshold;
    nms_threshold_ = nms_threshold;
    Boxes_.reserve(1000);
    center_priors_.reserve(2150);
    input_image_ = cv::Mat(input_height_, input_width_, CV_8UC3);
    // 生成锚点
    generate_grid_center_priors();
}






/**
 * @brief 执行目标检测和追踪
 * @param image 输入图像
 * @param boxes_res 输出检测结果
 * 
 * 完整处理流程：
 * 1. 图像预处理
 * 2. 模型推理
 * 3. 解码输出
 * 4. 非极大抑制
 * 5. 更新追踪器状态
 */
void NanoDet::detect(cv::Mat &image, std::vector<Box>& boxes_res) {
    // 1. 图像预处理 - 调整大小和归一化
    preprocess(image);
    
    // 2. 模型推理 - 通过OpenVINO运行Nanodet模型
    infer();
    
    // 3. 解码输出 - 将模型输出转换为检测框
    decode_infer();
    
    // 4. 非极大抑制 - 过滤重叠检测框
    NMS(boxes_res);
    
    // 5. 更新追踪器状态
    update_trackers(image, boxes_res);
    
    // 清空临时检测框缓存
    Boxes_.clear();
}

// 静态成员变量初始化
std::vector<KalmanTracker> NanoDet::trackers; // 当前活跃的追踪器列表
int NanoDet::next_track_id = 0;              // 下一个可用的追踪ID

/**
 * @brief 更新追踪器状态
 * @param image 用于可视化的图像
 * @param detections 当前帧的检测结果
 * 
 * 多目标追踪流程：
 * 1. 绘制预测框
 * 2. 预测所有追踪器状态
 * 3. 数据关联(检测框与追踪器匹配)
 * 4. 更新匹配的追踪器
 * 5. 为未匹配检测创建新追踪器
 * 6. 清理失效追踪器
 */
void NanoDet::update_trackers(cv::Mat& image, const std::vector<Box>& detections) {
    /**
     * IOU匹配阈值:
     * - 用于判断检测框与追踪预测框是否匹配
     * - 取值范围0-1，值越大匹配要求越严格
     * - 当两个框的IOU(交并比)大于此阈值时认为匹配成功
     * - 适当降低此值可以提高追踪鲁棒性，但会增加误匹配风险
     */
    const float IOU_THRESHOLD = 0.2f;
    
    /**
     * 最大允许丢失帧数:
     * - 控制追踪器在丢失目标后保持的帧数
     * - 当追踪器连续MAX_MISS_FRAMES帧未匹配到检测框时将被移除
     * - 增大此值可以使追踪在短暂遮挡后恢复，但会增加计算负担
     * - 减小此值可以快速清理失效追踪器，但可能丢失短暂遮挡的目标
     */
    const int MAX_MISS_FRAMES = 15;

    // // 1. 绘制预测框(半透明黄色)
    // for(auto& tracker : trackers) {
    //     if(tracker.time_since_update > 0) {
    //         Box predicted = tracker.get_state();
    //         cv::Scalar color(0, 255, 255, 128); // 半透明黄色
    //         cv::rectangle(image, 
    //                      cv::Point(predicted.x1, predicted.y1),
    //                      cv::Point(predicted.x2, predicted.y2),
    //                      color, 1);
    //     }
    // }

    // 2. 预测所有追踪器状态
    for(auto& tracker : trackers) {
        tracker.predict();
    }

    // 3. 数据关联 - IOU匹配(并行优化)
    std::vector<std::pair<int, int>> matches;
    std::vector<int> unmatched_trackers;
    std::vector<int> unmatched_detections(detections.size());
    std::iota(unmatched_detections.begin(), unmatched_detections.end(), 0);

    // 并行处理追踪器匹配
    #pragma omp parallel for schedule(dynamic)
    for(int i = 0; i < trackers.size(); i++) {
        // 跳过长时间未更新的追踪器
        if(trackers[i].time_since_update > MAX_MISS_FRAMES) {
            unmatched_trackers.push_back(i);
            continue;
        }

        // 寻找最佳匹配检测框
        float max_iou = 0;
        int best_match = -1;
        Box predicted_box = trackers[i].get_state();

        // 预筛选：仅计算中心点距离小于阈值的检测框
        for(int j = 0; j < detections.size(); j++) {
            // 计算中心点距离
            float center_dist = std::hypot(
                (predicted_box.x1 + predicted_box.x2)/2 - (detections[j].x1 + detections[j].x2)/2,
                (predicted_box.y1 + predicted_box.y2)/2 - (detections[j].y1 + detections[j].y2)/2);
            
            // 仅当中心点距离小于两框最大边长的1.5倍时才计算IOU
            if(center_dist < std::max(
                std::max(predicted_box.x2 - predicted_box.x1, predicted_box.y2 - predicted_box.y1),
                std::max(detections[j].x2 - detections[j].x1, detections[j].y2 - detections[j].y1)) * 1.5f) {
                
                float iou = calculate_iou(predicted_box, detections[j]);
                if(iou > max_iou && iou > IOU_THRESHOLD) {
                    max_iou = iou;
                    best_match = j;
                }
            }
        }

        // 处理匹配结果
        if(best_match != -1) {
            matches.emplace_back(i, best_match);
            // 从未匹配列表中移除已匹配检测
            unmatched_detections.erase(
                std::remove(unmatched_detections.begin(), 
                           unmatched_detections.end(), 
                           best_match),
                unmatched_detections.end());
        } else {
            unmatched_trackers.push_back(i);
        }
    }

    // 4. 更新匹配的追踪器
    for(auto& match : matches) {
        trackers[match.first].update(detections[match.second]);
    }

    // 5. 为未匹配检测创建新追踪器
    for(int idx : unmatched_detections) {
        KalmanTracker tracker(detections[idx]);
        tracker.id = next_track_id++;
        trackers.push_back(tracker);
    }

    // 6. 清理失效追踪器(长时间未更新)
    trackers.erase(
        std::remove_if(trackers.begin(), trackers.end(),
            [MAX_MISS_FRAMES](const KalmanTracker& t) {
                return t.time_since_update > MAX_MISS_FRAMES;
            }),
        trackers.end());
}

float NanoDet::calculate_iou(const Box& a, const Box& b) {
    float inter_x1 = std::max(a.x1, b.x1);
    float inter_y1 = std::max(a.y1, b.y1);
    float inter_x2 = std::min(a.x2, b.x2);
    float inter_y2 = std::min(a.y2, b.y2);
    
    float inter_area = std::max(0.f, inter_x2 - inter_x1) * std::max(0.f, inter_y2 - inter_y1);
    float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
    float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
    
    return inter_area / (area_a + area_b - inter_area);
}

void NanoDet::draw(cv::Mat &image, std::vector<Box> &boxes_res)
{
    for(auto & ibox : boxes_res){
        float left = ibox.x1;
        float top = ibox.y1;
        float right = ibox.x2;
        float bottom = ibox.y2;
        int class_label = ibox.label;
        float score = ibox.score;
        int track_id = ibox.track_id;
        
        // 根据追踪ID选择颜色
        cv::Scalar color;
        if(track_id >= 0) {
            std::tie(color[0], color[1], color[2]) = random_color(track_id);
        } else {
            std::tie(color[0], color[1], color[2]) = random_color(class_label);
        }

        // 绘制检测框
        cv::rectangle(image, cv::Point(left, top), cv::Point(right, bottom), color, 2);

        // 绘制标签信息
        std::string caption;
        if(track_id >= 0) {
            caption = cv::format("ID:%d %s %.2f", track_id, class_labels_[class_label].c_str(), score);
        } else {
            caption = cv::format("%s %.2f", class_labels_[class_label].c_str(), score);
        }
        
        int text_width = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
        cv::rectangle(image, cv::Point(left-3, top-33), cv::Point(left + text_width, top), color, -1);
        cv::putText(image, caption, cv::Point(left, top-5), 0, 1, cv::Scalar::all(0), 2, 16);

        // 处理追踪轨迹
        if(track_id >= 0) {
            cv::Point center((left + right)/2, (top + bottom)/2);
            
            // 获取或创建该ID的轨迹队列
            auto& track = tracking_history_[track_id];
            track.push_back(center);
            
            // 限制队列长度不超过200帧
            if(track.size() > TRACK_HISTORY_LENGTH) {
                track.pop_front();
            }
            
            // 绘制轨迹(使用直线段连接)
            if(track.size() > 1) {
                for(size_t i = 1; i < track.size(); i++) {
                    cv::line(image, track[i-1], track[i], color, 2, cv::LINE_AA);
                }
            }
        }
    }
}


// 调整大小和归一化
// 对图像进行resize和padding，通过双线性插值resize到模型要求的尺寸，padding以使图像中心与模型输入尺寸中心对齐
void NanoDet::preprocess(cv::Mat& image)
{
    // 通过双线性插值对图像进行resize
    float scale_x = (float)input_width_ / (float)image.cols;
    float scale_y = (float)input_height_ / (float)image.rows;
    float scale = std::min(scale_x, scale_y);

    // resize图像，源图像和目标图像几何中心的对齐
    i2d_[0] = scale;  i2d_[1] = 0;  i2d_[2] = (-scale * image.cols + input_width_ + scale  - 1) * 0.5;
    i2d_[3] = 0;  i2d_[4] = scale;  i2d_[5] = (-scale * image.rows + input_height_ + scale - 1) * 0.5;

    cv::Mat m2x3_i2d(2, 3, CV_32F, i2d_);  // image to dst(network), 2x3 matrix
    cv::Mat m2x3_d2i(2, 3, CV_32F, d2i_);  // dst to image, 2x3 matrix
    cv::invertAffineTransform(m2x3_i2d, m2x3_d2i);  // 计算一个反仿射变换

    // 对图像做平移缩放旋转变换，保持图像中心不变
    cv::warpAffine(image, input_image_, m2x3_i2d, input_image_.size(),
                   cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar::all(114));
//    cv::imshow("debug", input_image_);
//    cv::waitKey(0);
}


// 使用nanodet模型推理
void NanoDet::infer()
{
    // openvino 推理部分
    ov::element::Type input_type;
    switch (precision_) {
        case PRECISION_INT8:
            input_type = ov::element::u8;
            break;
        case PRECISION_FP16:
            input_type = ov::element::f16;
            break;
        case PRECISION_FP32:
            input_type = ov::element::f32;
            break;
        default:
            input_type = ov::element::u8;
            break;
    }
    // 模型需要NCHW格式输入 [1,3,416,416]
    ov::Shape input_shape = {1, 3, static_cast<size_t>(input_height_), static_cast<size_t>(input_width_)};

    // 创建临时缓冲区并转换数据布局
    cv::Mat nchw;
    cv::dnn::blobFromImage(input_image_, nchw);

    // 创建输入张量并设置数据
    ov::Tensor input_tensor = ov::Tensor(input_type, input_shape, nchw.data);

    // 设置推理请求的输入张量
    infer_request_.set_input_tensor(input_tensor);

    // 执行推理
    infer_request_.infer();

    // 得到输出特征张量
    output_tensor_ = infer_request_.get_output_tensor();

    // 获取输出数据指针
    output_ptr_ = output_tensor_.data<float>();
}

// 解码推理结果
void NanoDet::decode_infer()
{
    // 遍历所有中心点
    const int num_points = (int)center_priors_.size();
    const int num_channels = num_class_ + (reg_max_ + 1) * 4;

    for (int idx = 0; idx < num_points; idx++)
    {
        // 获取当前中心点的坐标和步长
        const int ct_x = center_priors_[idx].x;
        const int ct_y = center_priors_[idx].y;
        const int stride = center_priors_[idx].stride;

        // 获取当前中心点对应的输出特征
        float *ptr = output_ptr_ + idx * num_channels;
        // 获取当前中心点对应的类别和置信度
        int label = std::max_element(ptr, ptr + num_class_) - ptr;
        float score = ptr[label];

        // 如果置信度大于阈值
        if (score > score_threshold_)
        {
            // 获取当前中心点对应的bbox预测值
            float* bbox_pred = output_ptr_ + idx * num_channels + num_class_;
            // 将预测值解码成检测框
            Box box = disPred2Bbox(bbox_pred, label, score, ct_x, ct_y, stride);
            // 只添加有效Box
            if(box.label != -1) {
                Boxes_.emplace_back(box);
            } else {
                printf("[警告] 跳过无效检测框: 坐标(%.1f,%.1f)-(%.1f,%.1f)\n", 
                      box.x1, box.y1, box.x2, box.y2);
            }
        }
    }
}

// 解码单个中心点的预测结果
Box NanoDet::disPred2Bbox(float* dfl_det, int label, float score, int x, int y, int stride) const
{
    float ct_x = x * stride;
    float ct_y = y * stride;
    std::vector<float> dis_pred;
    dis_pred.reserve(4);
    float dis_after_sm[reg_max_ + 1];
    for (int i = 0; i < 4; i++)
    {
        float dis = 0;
        activation_function_softmax(dfl_det + i * (reg_max_ + 1), dis_after_sm, reg_max_ + 1);
        for (int j = 0; j < reg_max_ + 1; j++)
        {
            dis += j * dis_after_sm[j];
        }
        dis *= stride;
        dis_pred[i] = dis;
    }
    float xmin = (std::max)(ct_x - dis_pred[0], .0f);
    float ymin = (std::max)(ct_y - dis_pred[1], .0f);
    float xmax = (std::min)(ct_x + dis_pred[2], (float)input_width_);
    float ymax = (std::min)(ct_y + dis_pred[3], (float)input_height_);
    
    // 确保坐标有效性
    if(xmin >= xmax || ymin >= ymax) {
        return Box{0, 0, 0, 0, 0, -1}; // 返回无效Box
    }

    float image_x1 = d2i_[0] * xmin + d2i_[2];
    float image_y1 = d2i_[0] * ymin + d2i_[5];
    float image_x2 = d2i_[0] * xmax + d2i_[2];
    float image_y2 = d2i_[0] * ymax + d2i_[5];

    // 再次验证最终坐标
    if(!std::isfinite(image_x1) || !std::isfinite(image_y1) || 
       !std::isfinite(image_x2) || !std::isfinite(image_y2) ||
       image_x1 >= image_x2 || image_y1 >= image_y2) {
        return Box{0, 0, 0, 0, 0, -1}; // 返回无效Box
    }

    return Box{image_x1, image_y1, image_x2, image_y2, score, label};
}

// nms

/**
 * @brief 非极大值抑制(NMS)处理
 * @param boxes_res 输出参数，存储处理后的检测框结果
 * 
 * 实现步骤：
 * 1. 按置信度从高到低排序所有检测框
 * 2. 过滤掉低分检测框(score < 0.05)
 * 3. 对剩余检测框进行NMS处理，移除重叠度过高的框
 * 4. 限制最大输出检测框数量为100
 */
void NanoDet::NMS(std::vector<Box>& boxes_res)
{
    // 1. 预过滤：移除过大检测框(面积超过屏幕50%)、异常长宽比(>2)和低分检测框(score < 0.05)
    Boxes_.erase(std::remove_if(Boxes_.begin(), Boxes_.end(), 
        [this](const Box& b) { 
            float box_area = (b.x2 - b.x1) * (b.y2 - b.y1);
            float screen_area = input_width_ * input_height_;
            float width = b.x2 - b.x1;
            float height = b.y2 - b.y1;
            float ratio1 = width / height;
            float ratio2 = height / width;
            return (box_area / screen_area > 0.5f) || 
                   (ratio1 > 2.0f || ratio2 > 2.0f) || 
                   (b.score < 0.05f);
        }), 
        Boxes_.end());

    // 2. 按置信度从高到低排序所有检测框
    std::sort(Boxes_.begin(), Boxes_.end(), [](const Box& a, const Box& b) {
        return a.score > b.score; // 按score降序排列
    });

    // 3. 初始化标记数组和结果容器
    std::vector<bool> remove_flags(Boxes_.size(), false); // 标记是否移除该框
    boxes_res.reserve(std::min(Boxes_.size(), size_t(100))); // 预分配空间

    // IOU(Intersection over Union)计算函数
    // 用于计算两个检测框的重叠度
    auto iou = [](const Box& a, const Box& b){
        // 检查坐标有效性
        if(a.x1 >= a.x2 || a.y1 >= a.y2 || b.x1 >= b.x2 || b.y1 >= b.y2) 
            return 0.f; // 无效框返回0
        
        // 计算交叉区域坐标
        float cross_left = std::max(a.x1, b.x1);   // 交叉区域左边界
        float cross_top = std::max(a.y1, b.y1);     // 交叉区域上边界
        float cross_right = std::min(a.x2, b.x2);  // 交叉区域右边界
        float cross_bottom = std::min(a.y2, b.y2); // 交叉区域下边界

        // 检查交叉区域有效性
        if(cross_right <= cross_left || cross_bottom <= cross_top) 
            return 0.f; // 无交叉区域返回0
        
        // 计算交叉区域面积
        float cross_area = (cross_right - cross_left) * (cross_bottom - cross_top);
        // 计算两个框各自的面积
        float a_area = (a.x2 - a.x1) * (a.y2 - a.y1);
        float b_area = (b.x2 - b.x1) * (b.y2 - b.y1);
        // 计算并集面积
        float union_area = a_area + b_area - cross_area;
        
        // 防止除零和无效面积
        if(union_area <= 0 || cross_area <= 0) 
            return 0.f;
        
        // 返回IOU值(交叉面积/并集面积)
        return cross_area / union_area;
    };

    // 计算形状得分函数 (宽高比接近1的框得分更高)
    auto shape_score = [](const Box& box) {
        float width = box.x2 - box.x1;
        float height = box.y2 - box.y1;
        float ratio = std::min(width/height, height/width); // 计算宽高比
        return 0.5f * (1.0f + ratio); // 归一化到0.5-1.0
    };

    // 1. 计算距离惩罚系数
    auto distance_penalty = [](const Box& predicted, const Box& current) {
        // 计算预测框和当前框的中心点距离
        float pred_cx = (predicted.x1 + predicted.x2) / 2;
        float pred_cy = (predicted.y1 + predicted.y2) / 2;
        float curr_cx = (current.x1 + current.x2) / 2;
        float curr_cy = (current.y1 + current.y2) / 2;
        float distance = std::hypot(pred_cx - curr_cx, pred_cy - curr_cy);
        
        // 计算框对角线长度作为参考距离
        float box_size = std::hypot(current.x2 - current.x1, current.y2 - current.y1);
        
        // 增强惩罚力度: 
        // 1. 将参考距离从2倍改为1倍框尺寸
        // 2. 使用平方关系增加惩罚曲线陡峭度
        float normalized_dist = distance / box_size;  // 1倍框尺寸作为阈值
        float penalty = 1.0f - std::min(1.0f, normalized_dist * normalized_dist);
        return std::max(0.0f, penalty);
    };

    // 2. 计算尺寸惩罚系数(防止全屏检测框)
    auto size_penalty = [this](const Box& box) {
        // 计算检测框面积占屏幕面积的比例
        float box_area = (box.x2 - box.x1) * (box.y2 - box.y1);
        float screen_area = input_width_ * input_height_;
        float ratio = box_area / screen_area;
        
        // 当检测框超过屏幕面积30%时开始惩罚
        if (ratio > 0.3f) {
            // 使用指数衰减函数进行惩罚
            return expf(-5.0f * (ratio - 0.3f));
        }
        return 1.0f;
    };

    // 3. 寻找综合得分最高的框
    float max_score = 0;
    Box best_box;
    Box predicted_box = trackers.empty() ? Box{0,0,0,0,0,-1} : trackers[0].get_state();
    
    for(auto& box : Boxes_) {
        // 计算综合得分 = 置信度 * 形状得分 * 距离惩罚 * 尺寸惩罚
        float dist_penalty = trackers.empty() ? 1.0f : distance_penalty(predicted_box, box);
        float current_score = box.score * shape_score(box) * dist_penalty * size_penalty(box);
        
        if(current_score > max_score) {
            max_score = current_score;
            best_box = box;
        }
    }

    // 3. 更新或创建卡尔曼跟踪器
    if(max_score > 0) {
        if(trackers.empty()) {
            // 创建新跟踪器
            KalmanTracker tracker(best_box);
            tracker.id = next_track_id++;
            trackers.push_back(tracker);
        } else {
            // 更新现有跟踪器
            trackers[0].update(best_box);
        }
        best_box.track_id = trackers[0].id;
        boxes_res.push_back(best_box);
        
        // 预测下一帧位置
        trackers[0].predict();
    } else if(!trackers.empty()) {
        // 无检测框时仅预测
        trackers[0].predict();
    }

}


void NanoDet::generate_grid_center_priors()
{
    for (int stride : strides_)
    {
        int feat_w = std::ceil((float)input_width_ / (float)stride);
        int feat_h = std::ceil((float)input_height_ / (float)stride);
        for (int y = 0; y < feat_h; y++)
        {
            for (int x = 0; x < feat_w; x++)
            {
                CenterPrior ct{};
                ct.x = x;
                ct.y = y;
                ct.stride = stride;
                center_priors_.push_back(ct);
            }
        }
    }
}

void NanoDet::benchmark(int loop_num) {
    int warm_up = 50;
    input_image_ = cv::Mat(input_height_, input_width_, CV_8UC3, cv::Scalar(1, 1, 1));
    // warmup
    for (int i = 0; i < warm_up; i++)
    {
        infer();
    }
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < loop_num; i++)
    {
        infer();
    }
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    double time = 1000 * elapsed.count();
    printf("Average infer time = %.2f ms\n", time / loop_num);
}



