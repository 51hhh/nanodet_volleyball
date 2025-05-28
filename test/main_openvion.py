import cv2
import numpy as np
import time
from openvino.runtime import Core, get_version

print("OpenVINO:")
print(get_version())

class my_nanodet:
    def __init__(self, model_path, prob_threshold=0.4, iou_threshold=0.5):
        self.prob_threshold = prob_threshold
        self.iou_threshold = iou_threshold
        self.num_classes = 80
        self.input_shape = [640, 640]
        self.class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                            'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
                            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
                            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
                            'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
                            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
                            'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
        self.colors = [(255, 255, 0), (0, 255, 0), (0, 255, 255), (255, 0, 0)]
        self.num_classes = len(self.class_names)
        self.mlvl_anchors = [
            np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float32),
            np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float32),
            np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float32),
            np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float32),
            np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float32)
        ]
        self.strides = [8, 16, 32, 64, 128]
        self.reg_max = 7
        self.project = np.arange(self.reg_max + 1)
        self.softmax = lambda x, axis=1: np.exp(x) / np.expand_dims(np.sum(np.exp(x), axis=axis), axis)
        self.iou_threshold = 0.6
        self.prob_threshold = 0.3
        self.core = Core()
        self.model = self.core.read_model(model=model_path)
        self.compiled_model = self.core.compile_model(model=self.model, device_name="CPU")
        self.output_layer_names = [output_layer.get_names()[0] for output_layer in self.compiled_model.outputs]

    def resize_image(self, srcimg, size=[640, 640]):
        h, w = srcimg.shape[:2]
        scale = min(size[1] / w, size[0] / h)
        new_w, new_h = int(scale * w), int(scale * h)
        top, left = (size[0] - new_h) // 2, (size[1] - new_w) // 2
        resized_img = cv2.resize(srcimg, (new_w, new_h), interpolation=cv2.INTER_AREA)
        new_img = np.full((size[0], size[1], 3), 128, np.uint8)
        new_img[top:top + new_h, left:left + new_w] = resized_img
        return new_img, new_h, new_w, top, left

    def distance2bbox(self, points, distance, max_shape=None):
        x1 = points[:, 0] - distance[:, 0]
        y1 = points[:, 1] - distance[:, 1]
        x2 = points[:, 0] + distance[:, 2]
        y2 = points[:, 1] + distance[:, 3]
        if max_shape is not None:
            x1 = np.clip(x1, 0, max_shape[1])
            y1 = np.clip(y1, 0, max_shape[0])
            x2 = np.clip(x2, 0, max_shape[1])
            y2 = np.clip(y2, 0, max_shape[0])
        return np.stack([x1, y1, x2, y2], axis=-1)

    def post_process(self, preds, scale_factor=1, rescale=False):
        mlvl_bboxes = []
        mlvl_scores = []
        ind = 0
        for stride, anchors in zip(self.strides, self.mlvl_anchors):
            cls_score, bbox_pred = preds[ind:(ind + anchors.shape[0]), :self.num_classes], preds[
                                                                                               ind:(
                                                                                                       ind + anchors.shape[
                                                                                                           0]), self.num_classes:]
            ind += anchors.shape[0]
            bbox_pred = self.softmax(bbox_pred.reshape(-1, self.reg_max + 1), axis=1)
            # bbox_pred = np.sum(bbox_pred * np.expand_dims(self.project, axis=0), axis=1).reshape((-1, 4))
            bbox_pred = np.dot(bbox_pred, self.project).reshape(-1, 4)
            bbox_pred *= stride

            # nms_pre = cfg.get('nms_pre', -1)
            nms_pre = 1000
            if nms_pre > 0 and cls_score.shape[0] > nms_pre:
                max_scores = cls_score.max(axis=1)
                topk_inds = max_scores.argsort()[::-1][0:nms_pre]
                anchors = anchors[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                cls_score = cls_score[topk_inds, :]

            bboxes = self.distance2bbox(anchors, bbox_pred, max_shape=self.input_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(cls_score)

        mlvl_bboxes = np.concatenate(mlvl_bboxes, axis=0)
        if rescale:
            mlvl_bboxes /= scale_factor
        mlvl_scores = np.concatenate(mlvl_scores, axis=0)

        bboxes_wh = mlvl_bboxes.copy()
        bboxes_wh[:, 2:4] = bboxes_wh[:, 2:4] - bboxes_wh[:, 0:2]  ####xywh
        classIds = np.argmax(mlvl_scores, axis=1)
        confidences = np.max(mlvl_scores, axis=1)  ####max_class_confidence

        indices = cv2.dnn.NMSBoxes(bboxes_wh.tolist(), confidences.tolist(), self.prob_threshold,
                                    self.iou_threshold)

        if indices is not None:  # 增加判断indices是否为空的逻辑
            if isinstance(indices, tuple):
                if len(indices) > 0:
                    indices = indices[0]  # 提取元组中的第一个元素
                else:
                    print('NMSBoxes returned an empty tuple')
                    return np.array([]), np.array([]), np.array([])

            if len(indices) > 0:
                mlvl_bboxes = mlvl_bboxes[indices]
                confidences = confidences[indices]
                classIds = classIds[indices]
                return mlvl_bboxes, confidences, classIds
            else:
                print('nothing detect after NMS')
                return np.array([]), np.array([]), np.array([])
        else:
            print('NMSBoxes returned None')
            return np.array([]), np.array([]), np.array([])

    def detect(self, srcimg):
        img, newh, neww, top, left = self.resize_image(srcimg)
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)
        input_tensor = np.expand_dims(img, 0)
        # result = self.model.infer(inputs={self.input_node.any_name: input_tensor})
        results = self.compiled_model(input_tensor)[self.output_layer_names[0]][0]
        # print(results.shape)
        det_bboxes, det_conf, det_classid = self.post_process(results)
        return det_bboxes, det_conf, det_classid, srcimg, newh, neww, top, left

def process_video(video_path, model_path, conf_threshold, nms_threshold, save_result=False):
    net = my_nanodet(model_path, prob_threshold=conf_threshold, iou_threshold=nms_threshold)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video stream or file")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if save_result:
        output_path = "output.mp4"  # 你可以自定义输出路径
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    frame_count = 0  # 初始化帧数计数器
    start_time = time.time()  # 记录开始时间

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1  # 增加帧数计数器
            # 使用 my_nanodet 进行检测
            det_bboxes, det_conf, det_classid, srcimg, newh, neww, top, left = net.detect(frame)

            # 检查是否检测到任何物体
            if len(det_bboxes) > 0:
                ratioh, ratiow = srcimg.shape[0] / newh, srcimg.shape[1] / neww
                # 在图像上绘制检测结果
                for i in range(det_bboxes.shape[0]):
                    xmin, ymin, xmax, ymax = max(int((det_bboxes[i, 0] - left) * ratiow), 0), max(
                        int((det_bboxes[i, 1] - top) * ratioh), 0), min(
                        int((det_bboxes[i, 2] - left) * ratiow), srcimg.shape[1]), min(
                        int((det_bboxes[i, 3] - top) * ratioh),
                        srcimg.shape[0])
                    cv2.rectangle(srcimg, (xmin, ymin), (xmax, ymax), (0, 0, 255), thickness=1)
                    cv2.putText(srcimg,
                                net.class_names[det_classid[i]] + ': ' + str(round(det_conf[i], 3)),
                                (xmin, ymin - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), thickness=1)

            if save_result:
                out.write(srcimg)

            cv2.imshow('Video', srcimg)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # 释放资源
        cap.release()
        if save_result:
            out.release()
        cv2.destroyAllWindows()
        end_time = time.time()  # 记录结束时间
        total_time = end_time - start_time  # 计算总时间
        print(f"Processed {frame_count} frames in {total_time:.2f} seconds")
        if frame_count > 0:
            print(f"FPS: {frame_count / total_time:.2f}")  # 计算并打印FPS
        else:
            print("No frames were processed.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--modelpath', type=str, default='./nanodet.onnx', help='model path')
    parser.add_argument('--videopath', type=str, default='./rc.mp4', help='video path')
    parser.add_argument('--confThreshold', type=float, default=0.4, help='置信度阈值')
    parser.add_argument('--nmsThreshold', type=float, default=0.5, help='NMS阈值')
    parser.add_argument('--save_result', action='store_true', default=False, help='是否保存结果')
    args = parser.parse_args()

    process_video(args.videopath, args.modelpath, args.confThreshold, args.nmsThreshold, args.save_result)
