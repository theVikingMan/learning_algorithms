import pandas as pd
import numpy as np

from sklearn import model_selection, preprocessing, linear_model, metrics


def kfold(k: int, data: pd.DataFrame, SEED: int):
    folds = []
    randomized_data = data.sample(frac=1, random_state=SEED)
    n, m = randomized_data.shape
    fold_size = n // k

    for i in range(k):
        val_start = i * fold_size
        val_end = ((i + 1) * fold_size) if i != k - 1 else n

        X_train = np.concatenate((randomized_data[:val_start], randomized_data[val_end:]), axis=0)
        y_val = randomized_data[val_start:val_end]

        folds.append((X_train, y_val))

    return folds

def kfold_stratify(X, y, k, TEST_SIZE=0.2, SEED=42):
    folds = []

    for outer_i in range(k):
        X_train, X_test, y_train, y_test = model_selection.train_test_split(
            X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
        )
        validation_sets = []
        for inner_i in range(k):
            X_train_val, X_test_val, y_train_val, y_test_val = model_selection.train_test_split(
                X_train, y_train, test_size=TEST_SIZE, random_state=SEED, stratify=y_train
            )
            validation_sets.append(
                (X_train_val, X_test_val, y_train_val, y_test_val)
            )
        folds.append((validation_sets, X_test, y_test))

    return folds