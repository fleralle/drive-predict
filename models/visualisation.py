"""Visualisation utils librairy."""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def print_confusion_matrix(y_true, y_pred, labels=[0, 1]):
    """Print confusion matrix.

    Parameters
    ----------
    y_true : array
        Ground truth (correct) target values.
    y_pred : array
        Estimated targets as returned by a classifier.
    labels : array
        List of labels to index the matrix. This may be used to reorder or
        select a subset of labels. If none is given, those that appear at least
        once in y_true or y_pred are used in sorted order.

    """
    cnf_matrix = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots()
    tick_marks = np.arange(len(labels))

    # create heatmap
    sns.heatmap(cnf_matrix, annot=True, cmap="YlGnBu", fmt='g')
    ax.xaxis.set_label_position("top")
    plt.tight_layout()
    plt.title('Confusion matrix', y=1.1)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.xticks(tick_marks + .5, labels)
    plt.yticks(tick_marks + .5, labels)
