import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss, f1_score, r2_score, confusion_matrix, classification_report, roc_curve, auc


def draw_mask_plots(prediction, ground_truth, plot_file):
    plt.plot([x for x in range(len(prediction))], prediction, [x for x in range(len(prediction))], ground_truth)
    plt.savefig(plot_file)


def count_metrics_on_sample(prediction, ground_truth, json_file):
    with open(json_file, 'w') as f:
        json_dict = {}

        corr = np.corrcoef(prediction, ground_truth)[1, 1]  # get Pearson's correlation coefficient
        json_dict["corr"] = corr
        json.dump(json_dict, f)
        # need add more metrics


def draw_roc(pred_raw, pred_smooth, ground_truth, roc_curve_file):
    fpr = {}
    tpr = {}
    roc_auc = {}
    results = {"nn": pred_raw, "smooth_nn": pred_smooth}
    for k in results:
        fpr[k], tpr[k], _ = roc_curve(ground_truth, results[k])
        roc_auc[k] = auc(fpr[k], tpr[k])

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


class GroundTruthHardcoder:
    @staticmethod
    def get_epica():
        ratio = 100 / 96  # as 100 VGG embeddings are 96 seconds
        q_len = 1112  # length of the sample
        true_value = np.zeros(q_len, dtype=np.float32)
        true_value[int(11 * ratio):int(248 * ratio) + 1] = 1.0
        true_value[int(300 * ratio):int(524 * ratio) + 1] = 1.0
        true_value[int(587 * ratio):int(798 * ratio) + 1] = 1.0
        true_value[int(832 * ratio):int((17 * 60 + 25) * ratio) + 1] = 1.0
        return true_value

    @staticmethod
    def get_voice_india():
        ratio = 100 / 96  # as 100 VGG embeddings are 96 seconds
        q_len = 833  # length of the sample
        true_value = np.zeros(q_len, dtype=np.float32)
        true_value[int(50 * ratio):int(61 * ratio) + 1] = 1.0
        true_value[int(103 * ratio):int(110 * ratio) + 1] = 1.0
        true_value[int(195 * ratio):int(222 * ratio) + 1] = 1.0
        true_value[int(195 * ratio):int(222 * ratio) + 1] = 1.0
        true_value[int(264 * ratio):int(281 * ratio) + 1] = 1.0
        true_value[int(296 * ratio):int(302 * ratio) + 1] = 1.0
        true_value[int(312 * ratio):int(343 * ratio) + 1] = 1.0
        true_value[int(312 * ratio):int(343 * ratio) + 1] = 1.0
        true_value[int(374 * ratio):int(419 * ratio) + 1] = 1.0
        true_value[int(471 * ratio):int(495 * ratio) + 1] = 1.0
        true_value[int(639 * ratio):int(651 * ratio) + 1] = 1.0
        true_value[int(653 * ratio):int(709 * ratio) + 1] = 1.0
        true_value[int(711 * ratio):int(778 * ratio) + 1] = 1.0
        true_value[int(783 * ratio):] = 1.0
