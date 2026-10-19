import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp
import numpy as np

def crop_and_segment(segmentation_output, bbox_pred, original_size):
    """
    Mask the segmentation output outside the predicted bounding boxes and resize it to the original image size.
    """
    batch_size, _, h, w = segmentation_output.shape
    segmentation_final = torch.zeros_like(segmentation_output)

    for i in range(batch_size):
        bbox = bbox_pred[i].view(-1)
        x_min = int(bbox[0].item())
        y_min = int(bbox[1].item())
        x_max = int(bbox[2].item())
        y_max = int(bbox[3].item())

        x_min, x_max = max(0, x_min), min(w, x_max)
        y_min, y_max = max(0, y_min), min(h, y_max)
        
        # print(w,h,x_min, x_max,y_min,y_max)
        segmentation_final[i, :, y_min:y_max, x_min:x_max] = segmentation_output[i, :, y_min:y_max, x_min:x_max]

    return F.interpolate(segmentation_final, size=original_size, mode='bilinear', align_corners=True)

def confusion_matrix(x, y, n, ignore_label=None, mask=None):
    if mask is None:
        mask = np.ones_like(x) == 1
    k = (x >= 0) & (y < n) & (x != ignore_label) & (mask.astype(np.bool_))
    return np.bincount(n * x[k].astype(int) + y[k], minlength=n**2).reshape(n, n)

def getScores(conf_matrix):
    if conf_matrix.sum() == 0:
        return 0, 0, 0, 0, 0
    with np.errstate(divide='ignore',invalid='ignore'):
        globalacc = np.diag(conf_matrix).sum() / np.complex128(conf_matrix.sum()) # globalacc = np.diag(conf_matrix).sum() / np.cfloat(conf_matrix.sum())
        classpre = np.diag(conf_matrix) / conf_matrix.sum(0).astype(np.complex128)
        classrecall = np.diag(conf_matrix) / conf_matrix.sum(1).astype(np.complex128)
        IU = np.diag(conf_matrix) / (conf_matrix.sum(1) + conf_matrix.sum(0) - np.diag(conf_matrix)).astype(np.complex128)
        pre = classpre[1]
        recall = classrecall[1]
        iou = IU[1]
        aiu=(IU[1]+IU[0])/2
        F_score = 2*(recall*pre)/(recall+pre)
    return globalacc, pre, recall, F_score, iou, aiu

class DiceBCELoss(nn.Module):
    def __init__(self):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)      
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        weight=torch.zeros_like(targets)
        weight=torch.fill_(weight, 0.04)
        weight[targets>0]=0.96
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean', weight=weight)
        Dice_BCE = BCE + dice_loss
        
        return Dice_BCE
