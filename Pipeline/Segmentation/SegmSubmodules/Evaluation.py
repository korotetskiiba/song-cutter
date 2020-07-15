import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss, f1_score, r2_score, confusion_matrix, classification_report, roc_curve, auc, \
    roc_auc_score, precision_score, recall_score
import keras.backend as K


def draw_mask_plots(prediction, ground_truth, plot_file):
    """Draws plots of the predicted and ground truth mask and saves it to plot_file as image

    Args:
        prediction: the prediction binary mask vector
        ground_truth: the ground truth binary mask vector
        plot_file: path to file where plot will be saved"""
    # Have to add captions, tags and labels
    plt.figure()
    x = [x for x in range(len(prediction))]  # creating net to make plots
    plt.plot(x, prediction, label='prediction')
    plt.plot(x, ground_truth, label='ground truth')
    plt.xlabel('Time')
    plt.ylabel('Song mask')
    plt.title('Song mask plot')
    plt.legend(loc="lower right")
    plt.savefig(plot_file)


def count_metrics_on_sample(prediction, ground_truth, json_file, metrics_list=None):
    """Counts metrics for passed prediction and saves it to file in .JSON format. Default metrics:
    Pearson's correlation coefficient, f1-score, log_loss, roc_auc, precision, recall, intersection over union

    Args:
        prediction: the prediction binary mask vector
        ground_truth: the ground truth binary mask vector
        json_file: path to json file to save metrics
        metrics_list: list with custom metrics,
        metric function has 2 arguments: ground_truth and prediction. If None, default metrics used"""
    if metrics_list is None:  # assign default metrics
        metrics_list = [f1_score, log_loss, roc_auc_score, precision_score, recall_score, __intersection_over_union,
                        __corr]

    with open(json_file, 'w') as f:
        json_dict = {}  # init dictionary to save

        for metric in metrics_list:  # init
            json_dict[metric.__name__] = 0

        sample_cnt = prediction.shape[0]
        divisor = 0  # for normalization
        for current_num in range(sample_cnt):
            if K.sum(ground_truth[current_num,:] * prediction[current_num,:]) > 0:  # don't evaluate sample without song
                divisor += 1
                for metric in metrics_list:
                    json_dict[metric.__name__] += float(metric(ground_truth[current_num,:],
                                                               prediction[current_num,:]))

        for metric in metrics_list:  # normalize
            json_dict[metric.__name__] /= divisor

        json.dump(json_dict, f)  # save dictionary


def draw_roc(pred_raw, pred_smooth, ground_truth, roc_curve_file):
    """Draws plot of the ROC curve for the predictions (raw and smooth) and saves it to file as image

    Args:
        pred_raw: the prediction binary mask vector
        pred_smooth: the prediction binary mask vector after smoothing
        ground_truth: the ground truth binary mask vector
        roc_curve_file: path to the file where plot will be saved"""
    fpr = {}
    tpr = {}
    roc_auc = {}
    results = {"nn": pred_raw, "smooth_nn": pred_smooth}
    for k in results:  # calculate ROC & AUC
        fpr[k], tpr[k], _ = roc_curve(ground_truth, results[k])
        roc_auc[k] = auc(fpr[k], tpr[k])

    # draw plot
    plt.figure()
    for k in results:
        plt.plot(fpr[k], tpr[k],
                 lw=2, label='ROC curve (area = %0.2f)_{}'.format(k) % roc_auc[k])
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.savefig(roc_curve_file)


def __corr(true, pred):
    """Calculate Pearson's correlation coefficient

    Args:
        true: ground truth numpy tensor (binary mask)
        pred: prediction numpy tensor (binary mask)

    Returns:
        scalar (Pearson's correlation coefficient)"""
    return np.corrcoef(true, pred)[1, 0]


def __intersection_over_union(true, pred):
    """This is a function to calculate intersection-over-union metrics

    Args:
        true: ground truth numpy tensor (binary mask)
        pred: prediction numpy tensor (binary mask)

    Returns:
        scalar (intersection over union metrics value)"""
    intersection = true * pred
    true_inv = 1 - true
    union = true + (true_inv * pred)

    return K.sum(intersection) / K.sum(union)

