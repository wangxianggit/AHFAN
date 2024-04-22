import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, auc, precision_recall_curve
from sklearn.model_selection import train_test_split
import pickle

from config import *

data_dir = project_path / "dataset"


def aucPerformance(y_true, y_pred):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    roc_auc = roc_auc_score(y_true, y_pred)
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    auc_pr = auc(recall, precision)
    return roc_auc, auc_pr


