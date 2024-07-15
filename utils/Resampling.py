import pandas as pd
import numpy as np


def kfold(k: int, data: pd.DataFrame):
    folds = []
    SEED = 42
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