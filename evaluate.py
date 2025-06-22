""" Evaluate ROC
Returns:
    auc, eer: Area under the curve, Equal Error Rate
"""

# pylint: disable=C0103,C0301

##
# LIBRARIES
from __future__ import print_function

import os
from sklearn.metrics import roc_curve, auc, average_precision_score, f1_score, precision_score, recall_score
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
# rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=False)


def evaluate(labels, scores, metric='roc'):
    if metric == 'roc':
        return roc(labels, scores)
    elif metric == 'auprc':
        labels = labels.cpu()
        scores = scores.cpu()
        return auprc(labels, scores)
    elif metric == 'f1_score':
        labels = labels.cpu()
        scores = scores.cpu()
        threshold = 0.0126
        # scores[scores >= threshold] = 1
        # scores[scores <  threshold] = 0
        scores = torch.where(scores >= 0.22, torch.tensor(1), torch.tensor(0))
        return f1_score(labels, scores)
    elif metric == 'accuracy':
        labels = labels.cpu()
        scores = scores.cpu()
        scores = torch.where(scores >= 0.22, torch.tensor(1), torch.tensor(0))
        correct = torch.eq(labels, scores).float()
        acc = torch.mean(correct).item()
        return acc
    elif metric == 'all':
        auc_ = roc(labels, scores)
        labels = labels.cpu()
        scores = scores.cpu()
        auprc_ = auprc(labels, scores)
        scores = torch.where(scores >= 0.22, torch.tensor(1), torch.tensor(0))
        acc = torch.mean(torch.eq(labels, scores).float()).item()
        f1_score_ = f1_score(labels, scores)
        return [auc_, auprc_, acc, f1_score_]
    else:
        raise NotImplementedError("Check the evaluation metric.")


def roc(labels, scores, saveto=False):
    """Compute ROC curve and ROC area for each class"""
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    labels = labels.cpu()
    scores = scores.cpu()

    # True/False Positive Rates.
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    # Equal Error Rate
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)

    if saveto:
        plt.figure(figsize=(8, 8))
        lw = 3
        plt.plot(fpr, tpr, color='darkorange', lw=lw, label='(AUC = %0.3f, EER = %0.3f)' % (roc_auc, eer))
        plt.plot([eer], [1-eer], marker='o', markersize=5, color="cyan")
        plt.plot([0, 1], [1, 0], color='navy', lw=1, linestyle=':')
        plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        plt.xlim([0.0, 1.0])
        plt.grid(True)
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic (Mono)')
        plt.legend(loc="lower right")
        plt.show()
        # plt.savefig(os.path.join(saveto, "ROC.pdf"))
        # plt.close()

    return roc_auc

def auprc(labels, scores):
    ap = average_precision_score(labels, scores)
    return ap