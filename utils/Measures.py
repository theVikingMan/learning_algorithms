from typing import List

import pandas as pd
import numpy as np

def calc_confusion_matrix(y_true: List[int], y_predict:List[int]):
    TN = np.sum(np.logical_and(y_predict == 0, y_true == 0))
    TP = np.sum(np.logical_and(y_predict == 1, y_true == 1))
    FP = np.sum(np.logical_and(y_predict == 1, y_true == 0))
    FN = np.sum(np.logical_and(y_predict == 0, y_true == 1))

    confusion_matrix = {
        "TN": TN, "TP": TP, "FP": FP, "FN": FN
    }

    return confusion_matrix

def calc_accuracy(true_positive: int, false_positive: int, true_negative:int, false_negative: int):
    return (true_negative + true_negative) / (true_positive + true_negative + false_negative + false_positive)

def calc_recall(true_positive: int, false_negative: int):
    return (true_positive) / (true_positive + false_negative)

def calc_percision(true_positive: int, false_positive: int):
    return (true_positive) / (true_positive + false_positive)