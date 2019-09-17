import numpy as np


def max_iteration(matrix: np.array, num_snapshot: int) -> np.array:
    return np.max(matrix[:, num_snapshot, :], axis=0)


def min_iteration(matrix: np.array, num_snapshot: int) -> np.array:
    return np.min(matrix[:, num_snapshot, :], axis=0)


def median_iteration(matrix: np.array, num_snapshot: int) -> np.array:
    return np.median(matrix[:, num_snapshot, :], axis=0)


def remove_outliers(values: np.array) -> np.array:
    q1 = np.quantile(values, 0.25)
    q3 = np.quantile(values, 0.75)
    iqr = q3 - q1

    valid = np.where((values >= (q1 - 1.5 * iqr)) & (values <= (q3 + 1.5 * iqr)))
    return values[valid]
