from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np
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

if __name__ == '__main__':
    imgPredict = torch.FloatTensor([[1, 0.]])
    imgLabel = torch.FloatTensor([[0.]])

    acc = accuracy_classify(imgPredict, imgLabel, isTrain=False)

    print(acc)