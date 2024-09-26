import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import callbacks


def generate_boxes(dataset, batch, channel, conf_threshold_base, stride):
    boxes = list()
    
    for i, conf in enumerate(dataset[0][batch,:,0,channel]):
        if conf >= conf_threshold_base:
            c = (i + dataset[1][batch,i,0,0]) * stride
            w = np.exp(dataset[2][batch,i,0,0]) * stride

            ts = max(0, int(c - w/2))
            te = min(int(c + w/2), dataset[0].shape[1] * stride - 1)
            boxes.append([ts, te, conf])

    return np.array(boxes)


def get_ious(boxes_gt, boxes_pred):
    boxes_gt_l, boxes_gt_r = np.split(boxes_gt[:,:2], 2, axis=1)
    boxes_pred_l, boxes_pred_r = np.split(boxes_pred[:,:2], 2, axis=1)
    
    boxes_gt_width = boxes_gt[:, 1] - boxes_gt[:, 0]
    boxes_pred_width = boxes_pred[:, 1] - boxes_pred[:, 0]
    
    intersect_l = np.maximum(boxes_gt_l, boxes_pred_l.transpose())
    intersect_r = np.minimum(boxes_gt_r, boxes_pred_r.transpose())
    intersect = np.maximum(intersect_r - intersect_l, 0)
    union = boxes_gt_width[..., np.newaxis] + boxes_pred_width[np.newaxis, ...] - intersect
    
    return np.divide(intersect, union, out=np.zeros_like(intersect, dtype=float), where=union != 0)


def get_matches(ious, iou_threshold):
    num_gt = ious.shape[0]
    num_pred = ious.shape[1]
    
    matches_gt = -1 * np.ones(num_gt, dtype=int)
    matches_pred = -1 * np.ones(num_pred, dtype=int)
    
    for pred_idx in range(num_pred):
        match_gt_idx = -1
        iou = np.minimum(iou_threshold, 1. - keras.backend.epsilon())
        
        for gt_idx in range(num_gt):
            if matches_gt[gt_idx] != -1:
                continue
    
            if ious[gt_idx, pred_idx] < iou:
                continue
    
            iou = ious[gt_idx, pred_idx]
            match_gt_idx = gt_idx
    
        matches_pred[pred_idx] = match_gt_idx
        if match_gt_idx != -1:
            matches_gt[match_gt_idx] = pred_idx

    return matches_gt, matches_pred


def get_lp(boxes_gt, boxes_pred, matches_gt, matches_pred):
    labels = np.append(np.ones(boxes_gt.shape[0]), np.zeros(np.count_nonzero(matches_pred == -1)))
    predictions = np.concatenate((boxes_pred[matches_gt[matches_gt != -1], 2], np.zeros(np.count_nonzero(matches_gt == -1)),
                                  boxes_pred[matches_pred == -1, 2]))
    return labels, predictions


def average_precision(labels, predictions):
    lp = np.stack((labels, predictions), axis=-1)
    lp_argsort = np.argsort(-lp[:,1], kind='mergesort')
    lp = lp[lp_argsort]
    
    pr = list()
    
    tp = 0
    fp = 0
    fn = np.count_nonzero(lp[:,0] == 1.)
    for l, _ in lp[lp[:,1] != 0.]:
        if l == 1.:
            tp += 1
            fn -= 1
        else:
            fp += 1
            
        precision = tp / (tp + fp) if (tp + fp) > 0 else float('nan')
        recall = tp / (tp + fn) if (tp + fn) > 0 else float('nan')

        if pr and recall == pr[-1][1]:
            if precision > pr[-1][0]:
                pr[-1][0] = precision
            pr[-1][2] += 1
        else:
            pr.append([precision, recall, 1])

    p_interp = 0.
    f_total = 0.
    total = 0.
    for p, _, f in reversed(pr):
        p_interp = max(p, p_interp)

        total += p_interp * f
        f_total += f

    return total / f_total if f_total > 0 else float('nan')


# @tf.function
def mAP(out_dataset, target_dataset, stride, conf_threshold_base, iou_threshold):
    out_dataset_shape = out_dataset[0].shape
    ap_shape = (out_dataset_shape[0], out_dataset_shape[3])
    ap_array = np.empty((ap_shape))

    for i in np.ndindex(ap_shape):
        print("            \r{}".format(i), end="\r")
        boxes_gt = generate_boxes(target_dataset, *i, 1., stride)
        boxes_pred = generate_boxes(out_dataset, *i, conf_threshold_base, stride)
        
        if boxes_gt.size == 0:
            if boxes_pred.size == 0:
                ap_array[i] = float("nan")
            else:
                ap_array[i] = 0.

            continue

        if boxes_pred.size == 0:
            ap_array[i] = 0.
            continue 

        boxes_pred_argsort = np.argsort(-boxes_pred[:,2], kind='mergesort')
        boxes_pred = boxes_pred[boxes_pred_argsort]
                
        ious = get_ious(boxes_gt, boxes_pred)
        matches_gt, matches_pred = get_matches(ious, iou_threshold)
        labels, predictions = get_lp(boxes_gt, boxes_pred, matches_gt, matches_pred)

        ap_array[i] = average_precision(labels, predictions)
        
    return np.nanmean(np.nanmean(ap_array, axis=1))


class MeanAveragePrecision(callbacks.Callback):
    def __init__(self, input_dataset, target_dataset, stride, conf_threshold_base, iou_threshold):
        super().__init__()

        self.input_dataset = input_dataset
        self.target_dataset = target_dataset
        self.stride = stride
        self.conf_threshold_base = conf_threshold_base
        self.iou_threshold = iou_threshold

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 30 == 0:
            out_dataset = self.model.predict((self.input_dataset[0], self.input_dataset[1]))
            map = mAP(out_dataset, self.target_dataset, self.stride, self.conf_threshold_base, self.iou_threshold)

            print("[{}] mAP: {}".format(epoch + 1, map))

