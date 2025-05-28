import cv2
import numpy as np
import argparse
import onnxruntime as ort
import math
import time
import os

class my_nanodet():
    def __init__(self, model_pb_path, prob_threshold=0.4, iou_threshold=0.3):
        self.classes = ['volleyball']
        self.num_classes = len(self.classes)
        self.prob_threshold = prob_threshold
        self.iou_threshold = iou_threshold
        ### normalize: [[103.53, 116.28, 123.675], [57.375, 57.12, 58.395]]
        self.mean = np.array([103.53, 116.28, 123.675], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([57.375, 57.12, 58.395], dtype=np.float32).reshape(1, 1, 3)
        so = ort.SessionOptions()
        so.log_severity_level = 3
        self.net = ort.InferenceSession(model_pb_path, so)
        self.input_shape = (self.net.get_inputs()[0].shape[2], self.net.get_inputs()[0].shape[3])
        self.reg_max = int((self.net.get_outputs()[0].shape[-1] - self.num_classes) / 4) - 1
        self.project = np.arange(self.reg_max + 1)
        self.strides = (8, 16, 32, 64)
        self.mlvl_anchors = []
        for i in range(len(self.strides)):
            anchors = self._make_grid(
                (math.ceil(self.input_shape[0] / self.strides[i]), math.ceil(self.input_shape[1] / self.strides[i])),
                self.strides[i])
            self.mlvl_anchors.append(anchors)
        self.keep_ratio = False

    def _make_grid(self, featmap_size, stride):
        feat_h, feat_w = featmap_size
        shift_x = np.arange(0, feat_w) * stride
        shift_y = np.arange(0, feat_h) * stride
        xv, yv = np.meshgrid(shift_x, shift_y)
        xv = xv.flatten()
        yv = yv.flatten()
        return np.stack((xv, yv), axis=-1)
        # cx = xv + 0.5 * (stride - 1)
        # cy = yv + 0.5 * (stride - 1)
        # return np.stack((cx, cy), axis=-1)

    def softmax(self, x, axis=1):
        x_exp = np.exp(x)
        # 如果是列向量，则axis=0
        x_sum = np.sum(x_exp, axis=axis, keepdims=True)
        s = x_exp / x_sum
        return s

    def _normalize(self, img):
        img = img.astype(np.float32)
        # img = (img / 255.0 - self.mean / 255.0) / (self.std / 255.0)
        img = (img - self.mean) / (self.std)
        return img

    def resize_image(self, srcimg, keep_ratio=True):
        top, left, newh, neww = 0, 0, self.input_shape[0], self.input_shape[1]
        if keep_ratio and srcimg.shape[0] != srcimg.shape[1]:
            hw_scale = srcimg.shape[0] / srcimg.shape[1]
            if hw_scale > 1:
                newh, neww = self.input_shape[0], int(self.input_shape[1] / hw_scale)
                img = cv2.resize(srcimg, (neww, newh), interpolation=cv2.INTER_AREA)
                left = int((self.input_shape[1] - neww) * 0.5)
                img = cv2.copyMakeBorder(img, 0, 0, left, self.input_shape[1] - neww - left, cv2.BORDER_CONSTANT,
                                         value=0)  # add border
            else:
                newh, neww = int(self.input_shape[0] * hw_scale), self.input_shape[1]
                img = cv2.resize(srcimg, (neww, newh), interpolation=cv2.INTER_AREA)
                top = int((self.input_shape[0] - newh) * 0.5)
                img = cv2.copyMakeBorder(img, top, self.input_shape[0] - newh - top, 0, 0, cv2.BORDER_CONSTANT, value=0)
        else:
            img = cv2.resize(srcimg, self.input_shape, interpolation=cv2.INTER_AREA)
        return img, newh, neww, top, left

    def post_process(self, preds, scale_factor=1, rescale=False):
        mlvl_bboxes = []
        mlvl_scores = []
        ind = 0
        for stride, anchors in zip(self.strides, self.mlvl_anchors):
            cls_score, bbox_pred = preds[ind:(ind + anchors.shape[0]), :self.num_classes], preds[ind:(ind + anchors.shape[0]), self.num_classes:]
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

        indices = cv2.dnn.NMSBoxes(bboxes_wh.tolist(), confidences.tolist(), self.prob_threshold, self.iou_threshold)

        if indices is not None: # 增加判断indices是否为空的逻辑
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

    def detect(self, srcimg):
        img, newh, neww, top, left = self.resize_image(srcimg, keep_ratio=self.keep_ratio)
        img = self._normalize(img)
        blob = np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0)

        outs = self.net.run(None, {self.net.get_inputs()[0].name: blob})[0].squeeze(axis=0)
        det_bboxes, det_conf, det_classid = self.post_process(outs)
        return det_bboxes, det_conf, det_classid, srcimg,newh, neww, top, left

        # results = []
        ratioh, ratiow = srcimg.shape[0] / newh, srcimg.shape[1] / neww
        for i in range(det_bboxes.shape[0]):
            xmin, ymin, xmax, ymax = max(int((det_bboxes[i, 0] - left) * ratiow), 0), max(
                int((det_bboxes[i, 1] - top) * ratioh), 0), min(
                int((det_bboxes[i, 2] - left) * ratiow), srcimg.shape[1]), min(int((det_bboxes[i, 3] - top) * ratioh),
                                                                               srcimg.shape[0])
            # results.append((xmin, ymin, xmax, ymax, self.classes[det_classid[i]], det_conf[i]))
            cv2.rectangle(srcimg, (xmin, ymin), (xmax, ymax), (0, 0, 255), thickness=1)
            print(self.classes[det_classid[i]] + ': ' + str(round(det_conf[i], 3)))
            cv2.putText(srcimg, self.classes[det_classid[i]] + ': ' + str(round(det_conf[i], 3)), (xmin, ymin - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), thickness=1)
        cv2.imwrite('result.jpg', srcimg)
        return srcimg

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
            det_bboxes, det_conf, det_classid,srcimg, newh, neww, top, left = net.detect(frame)
            # 检查是否检测到任何物体
            if len(det_bboxes) > 0:
                ratioh, ratiow = srcimg.shape[0] / newh, srcimg.shape[1] / neww
                # 在图像上绘制检测结果
                for i in range(det_bboxes.shape[0]):
                    xmin, ymin, xmax, ymax = max(int((det_bboxes[i, 0] - left) * ratiow), 0), max(
                    int((det_bboxes[i, 1] - top) * ratioh), 0), min(
                    int((det_bboxes[i, 2] - left) * ratiow), srcimg.shape[1]), min(int((det_bboxes[i, 3] - top) * ratioh),
                                                                                   srcimg.shape[0])
                    cv2.rectangle(srcimg, (xmin, ymin), (xmax, ymax), (0, 0, 255), thickness=1)
                    cv2.putText(srcimg, net.classes[det_classid[i]] + ': ' + str(round(det_conf[i], 3)), (xmin, ymin - 10),
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
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--videopath', type=str, default='demo.mp4', help="video path")
    parser.add_argument('--modelpath', type=str, default='nanodet.onnx', help="onnx filepath")
    parser.add_argument('--confThreshold', default=0.4, type=float, help='class confidence')
    parser.add_argument('--nmsThreshold', default=0.6, type=float, help='nms iou thresh')
    parser.add_argument('--save_result', action='store_true', help='whether to save the inference result of video')
    args = parser.parse_args()

    process_video(args.videopath, args.modelpath, args.confThreshold, args.nmsThreshold, args.save_result)
