import itertools
import os
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report
import numpy as np
import matplotlib.pyplot as plt


def __plot_confusion_matrix(cm, classes, path, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure(figsize=(10, 10))
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    name = "cnf_normalized.png" if normalize else "cnf_not_normalized.png"
    plt.savefig(os.path.join(path, name))


def cnf_matrices(pred_labels, target_labels, category_dict, path):
    """
    Calculates, plots and saves to path normalized and non-normalized confusion matrices.

    :param pred_labels: labels predicted by the model;
    :param target_labels: ground truth labels;
    :param category_dict: dict{<number>:<genre>};
    :param path: path to dir where png's of matrices are to be saved;
    :return: void.
    """
    cnf_matrix = confusion_matrix(target_labels, pred_labels)
    class_list = [category_dict[v] for v in category_dict if category_dict[v] != 'none']

    # Plot non-normalized confusion matrix
    plt.figure()
    __plot_confusion_matrix(
        cnf_matrix,
        classes=class_list,
        title='Confusion matrix, without normalization',
        path=path
    )

    # Plot normalized confusion matrix
    plt.figure()
    __plot_confusion_matrix(
        cnf_matrix,
        classes=class_list,
        normalize=True,
        title='Normalized confusion matrix',
        path=path
    )


def count_metrics_on_sample(pred_labels, target_labels):
    """
    Creates classification report.

    :param pred_labels: labels predicted by the model;
    :param target_labels: ground truth labels;
    :return: void.
    """
    print("Accuracy:" + str(accuracy_score(target_labels, pred_labels)))
    print(classification_report(target_labels, pred_labels))


