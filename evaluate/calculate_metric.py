from evaluate.binary_confusion_matrix import get_binary_confusion_matrix
from evaluate.binary_statistical_metrics import get_accuracy, get_true_positive_rate, get_true_negative_rate, get_precision, get_f1_socre, get_iou
# from util.numpy_utils import tensor2numpy

def calculate_metrics(preds, targets,device):
    curr_TP, curr_FP, curr_TN, curr_FN = get_binary_confusion_matrix(
        input_=preds, target=targets, device =device, pixel = None, 
        threshold=0.5,
        reduction='sum',a=1)
    curr_acc = get_accuracy(true_positive=curr_TP,
                            false_positive=curr_FP,
                            true_negative=curr_TN,
                            false_negative=curr_FN)
    curr_recall = get_true_positive_rate(true_positive=curr_TP,
                                         false_negative=curr_FN)
    curr_precision = get_precision(true_positive=curr_TP,
                                   false_positive=curr_FP)
    curr_f1_score = get_f1_socre(true_positive=curr_TP,
                                 false_positive=curr_FP,
                                 false_negative=curr_FN)
    curr_iou = get_iou(true_positive=curr_TP,
                       false_positive=curr_FP,
                       false_negative=curr_FN)
    
    curr_TP, curr_FP, curr_TN, curr_FN = get_binary_confusion_matrix(
        input_=preds, target=targets, device =device, pixel = None, 
        threshold=0.5,
        reduction='sum',a=2)
    curr_iou1 = get_iou(true_positive=curr_TP,
                       false_positive=curr_FP,
                       false_negative=curr_FN)
    curr_aiu=(curr_iou+curr_iou1)/2
    return (curr_acc, curr_recall,curr_precision,
            curr_f1_score, curr_iou, curr_aiu)


def calculate_metrics_from_counts(true_positive, false_positive, true_negative, false_negative):
    """Calculate dataset-level metrics from accumulated scalar tensor counts.

    Return accuracy, recall, precision, F1, crack IoU and mean crack/background IoU.
    """
    accuracy = get_accuracy(true_positive, false_positive, true_negative, false_negative)
    recall = get_true_positive_rate(true_positive, false_negative)
    precision = get_precision(true_positive, false_positive)
    f1 = get_f1_socre(true_positive, false_positive, false_negative)
    iou = get_iou(true_positive, false_positive, false_negative)
    background_iou = get_iou(true_negative, false_negative, false_positive)
    return accuracy, recall, precision, f1, iou, (iou + background_iou) / 2
