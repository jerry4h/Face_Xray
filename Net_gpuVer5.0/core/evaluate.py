from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np
from sklearn import metrics
import torch.nn.functional as F
from torch.autograd import Variable

def pixelAccuracy(confusionMatrix):
    # return all class overall pixel accuracy
    # acc = (TP + TN) / (TP + TN + FP + TN)
    acc = np.diag(confusionMatrix).sum() / confusionMatrix.sum()
    return acc

def meanIntersectionOverUnion(confusionMatrix):
    # Intersection = TP Union = TP + FP + FN
    # IoU = TP / (TP + FP + FN)
    intersection = np.diag(confusionMatrix)
    union = np.sum(confusionMatrix, axis=1) + np.sum(confusionMatrix, axis=0) - np.diag(
        confusionMatrix)
    IoU = intersection / union
    mIoU = np.nanmean(IoU)
    return mIoU

def genConfusionMatrix(imgPredict, imgLabel, numClass):

    # remove classes from unlabeled pixels in gt image and predict
    mask = (imgLabel >= 0) & (imgLabel < numClass)
    label = numClass * imgLabel[mask] + imgPredict[mask]
    count = np.bincount(label, minlength=numClass ** 2)
    confusionMatrix = count.reshape(numClass, numClass)
    return confusionMatrix


def accuracy_PA(label_preds, label_trues):

    label_preds[label_preds >= 0.5] = 1
    label_preds[label_preds < 0.5] = 0
    label_trues[label_trues >= 0.5] = 1
    label_trues[label_trues < 0.5] = 0

    # Variable -> Tensor -> numpy -> int
    label_preds = label_preds.cpu().data.numpy().astype(int)
    label_trues = label_trues.cpu().data.numpy().astype(int)

    # 0,1
    numClass = 2
    with torch.no_grad():
        confusionMatrix = np.zeros((numClass,) * 2)

        assert label_preds.shape == label_trues.shape
        confusionMatrix += genConfusionMatrix(label_preds, label_trues, numClass)

        acc = pixelAccuracy(confusionMatrix)
        mIoU =  meanIntersectionOverUnion(confusionMatrix)

    return acc, mIoU

def accuracy_classify(label_preds, label_trues, isTrain=True):

    # 训练时softmax不起作用，故补充
    if isTrain:
        # 取概率最高者的下标
        prediction = torch.max(F.softmax(label_preds, dim=1), 1)[1]
    else:
        prediction = torch.max(label_preds, 1)[1]

    accuracy = float(sum(prediction == label_trues)) / len(label_trues)

    return accuracy


# Scikit-learn metrics

def find_best_threshold(fpr, tpr):
    diff = np.array(tpr) - np.array(1.-fpr)
    diff = diff**2
    index = np.argmin(diff)
    return index


def find_nearest_fpr(fpr, fpr_target):
    diff = np.array(fpr) - fpr_target
    diff = diff**2
    index = np.argmin(diff)
    return index

def evaluate(y_label, y_score, pos_label=1):
    fpr, tpr, threshold = metrics.roc_curve(y_label, y_score, pos_label=pos_label)
    auc = metrics.auc(fpr, tpr)
    index = find_best_threshold(fpr, tpr)
    acc_0 = tpr[index]
    acc_1 = 1. - fpr[index]

    return acc_0, acc_1, threshold[index], auc


def evaluate_at_fpr(y_label, y_score, pos_label=1, fpr_target=0.1):
    fpr, tpr, threshold = metrics.roc_curve(y_label, y_score, pos_label=pos_label)
    auc = metrics.auc(fpr, tpr)
    index = find_nearest_fpr(fpr, fpr_target)
    tpr_target = tpr[index]
    fpr_nearest = fpr[index]

    return tpr_target, fpr_nearest, threshold[index], auc


def get_confusion_matrix(y_true, y_pred):
    return metrics.confusion_matrix(y_true, y_pred)

def classify_evaluate_roc(y_labels, y_scores, needROC=False):
    """y_labels, y_scores torch.Tensor 或 np.array 形式，[N] 的尺寸，一般用 pos 的概率表示
    """
    if isinstance(y_labels, np.ndarray):
        pass
    elif isinstance(y_labels, torch.Tensor):
        y_labels = y_labels.clone().cpu().numpy()
    else:
        raise NotImplementedError()
    
    if isinstance(y_scores, np.ndarray):
        pass
    elif isinstance(y_scores, torch.Tensor):
        y_scores = y_scores.clone().cpu().numpy()
    else:
        raise NotImplementedError()
    
    pred = y_scores > 0.5

    confusionMatrix = get_confusion_matrix(y_labels, pred)
    # ACC = (confusionMatrix[0,0] + confusionMatrix[1,1]) / confusionMatrix.sum()
    ACC = np.diag(confusionMatrix).sum() / confusionMatrix.sum()
    if needROC:
        acc_0, acc_1, thre, AUC = evaluate(y_labels, y_scores, pos_label=1)
        EER = 1 - (acc_0 + acc_1) * 0.5
    else:
        EER, AUC, thre = -1, -1, -1
        
    return ACC, EER, AUC, thre, confusionMatrix


if __name__ == '__main__':
    imgPredict = torch.FloatTensor([[1, 0.]])
    imgLabel = torch.FloatTensor([[0.]])

    acc = accuracy_classify(imgPredict, imgLabel, isTrain=False)

    print(acc)