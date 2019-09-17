import pandas
import numpy as np

from typing import List, Callable
from scipy.stats import mannwhitneyu, kruskal

from sources.experiment.experiment_loader import ExperimentLoader


def save_mni_table_csv(csv_path: str, experiments_list: List[ExperimentLoader], labels: List[str]):
    dict_list = {}
    n_snp = experiments_list[0].dataset['snapshot_count']

    for i_exp, exp in enumerate(experiments_list):
        snp_values = []

        mni_matrix = exp.get_mni_matrix()

        for i in range(n_snp):
            mean = np.mean(mni_matrix[:, i])
            std = np.std(mni_matrix[:, i])

            snp_values.append("{0: .4f} +/- {1: .4f}".format(mean, std))

        dict_list[labels[i_exp]] = snp_values

    df = pandas.DataFrame.from_dict(dict_list, orient='index')
    df.to_csv(csv_path)
    print(df)


def save_mni_kruskall_table_csv(csv_path: str, experiments_list: List[ExperimentLoader], alpha=0.01):

    n_snp = experiments_list[0].dataset['snapshot_count']
    mni_exp_list = [exp.get_mni_matrix() for exp in experiments_list]

    empty_str = ""
    for i in range(n_snp):

        data = [nmi[:, i] for nmi in mni_exp_list]
        try:
            _, p = kruskal(*data)

            if p <= alpha:
                # reject the null hypothesis, are not the same
                empty_str += "\u2714"
            else:
                # cannot reject the null hypothesis
                empty_str += "\u2716"
        except ValueError:
            # if all the values are the same then don't reject the null hypothesis
            empty_str += "\u2592"

    with open(csv_path, 'w') as f:
        f.write(empty_str)

    print(empty_str)


# --------------------------------------------------------------------------------------------------------------------
# AVAILABLE DATA
# ---------------------------------------------------------------------------------------------------------------------

def _get_max_sum_hv_data(exp_list: List[ExperimentLoader]) -> List[np.array]:
    hv_matrix = [exp.get_hypervolume_matrix() for exp in exp_list]
    hv_matrix = [hv[:, :, -1] for hv in hv_matrix]
    hv_sum = [np.sum(hv, axis=1) for hv in hv_matrix]
    return hv_sum


def _get_avg_sqr_error_data(exp_list: List[ExperimentLoader]) -> List[np.array]:
    mni_list = [exp.get_mni_matrix() for exp in exp_list]
    sqr_error = [((mni - 1.0) ** 2).mean(axis=1) for mni in mni_list]
    return sqr_error

# --------------------------------------------------------------------------------------------------------------------
# GENERIC TABLE GENERATION
# ---------------------------------------------------------------------------------------------------------------------


def _save_default_matrix_csv(csv_path: str, experiments_matrix: List[List[ExperimentLoader]], labels: List[str],
                             datasets: List[str], data_method: Callable[[List[ExperimentLoader]], List[np.array]],
                             prec: int):

    assert len(experiments_matrix) == len(labels), "first dimension of experiment matrix must have the same length as labels"
    assert len(experiments_matrix[0]) == len(datasets), "second dimension of experiment matrix must have the same length as datasets"

    dict_list = {}
    n_exp = len(datasets)

    for i_exp, exp_list in enumerate(experiments_matrix):
        hv_sum = data_method(exp_list)
        mean = [np.mean(hv) for hv in hv_sum]
        std = [np.std(hv) for hv in hv_sum]

        snp_values = ["{: .{}f} +/- {: .{}f}".format(mean[i], prec, std[i], prec) for i in range(n_exp)]

        dict_list[labels[i_exp]] = snp_values

    df = pandas.DataFrame.from_dict(dict_list, orient='index', columns=datasets)
    df.to_csv(csv_path)
    print(df)


def _save_default_hypothesis_csv(csv_path: str, experiments_matrix: List[List[ExperimentLoader]],
                                 labels: List[str], datasets: List[str], alpha: float,
                                 data_method: Callable[[List[ExperimentLoader]], List[np.array]]):

    assert len(experiments_matrix) == len(labels), "first dimension of experiment matrix must have the same length as labels"
    assert len(experiments_matrix[0]) == len(datasets), "first dimension of experiment matrix must have the same length as labels"

    dict_list = {}

    n_exp = len(labels)
    n_dat = len(datasets)

    empty_str = ""
    for i in range(n_dat):
        empty_str += "\u2592"

    for i in range(n_exp):
        row = [empty_str]*n_exp

        for j in range(i+1, n_exp):
            actual_str = ""

            data_1 = data_method(experiments_matrix[i])
            data_2 =  data_method(experiments_matrix[j])

            for i_dats in range(n_dat):
                try:
                    _, p = mannwhitneyu(data_1[i_dats], data_2[i_dats])

                    if p <= alpha:
                        # reject the null hypothesis, are not the same
                        actual_str += "\u2714"
                    else:
                        # cannot reject the null hypothesis
                        actual_str += "\u2716"
                except ValueError:
                    # if all the values are the same then don't reject the null hypothesis
                    actual_str += "\u2716"

            row[j] = actual_str

        dict_list[labels[i]] = row

    df = pandas.DataFrame.from_dict(dict_list, orient='index')
    df.columns = labels
    df.to_csv(csv_path)

# --------------------------------------------------------------------------------------------------------------------
# SPECIFIC TABLE GENERATION
# ---------------------------------------------------------------------------------------------------------------------


def save_max_sum_hv_table_csv(csv_path: str, experiments_matrix: List[List[ExperimentLoader]], labels: List[str], datasets: List[str]):
    _save_default_matrix_csv(csv_path, experiments_matrix, labels, datasets, _get_max_sum_hv_data, 2)


def save_max_sum_hv_hypothesis_table_csv(csv_path: str, experiments_matrix: List[List[ExperimentLoader]],
                                         labels: List[str], datasets: List[str], alpha=0.01):
    _save_default_hypothesis_csv(csv_path, experiments_matrix, labels, datasets, alpha, _get_max_sum_hv_data)


def save_avg_sqr_error_table_csv(csv_path: str, experiments_matrix: List[List[ExperimentLoader]], labels: List[str], datasets: List[str]):
    _save_default_matrix_csv(csv_path, experiments_matrix, labels, datasets, _get_avg_sqr_error_data, 4)


def save_avg_sqr_error_hypothesis_table_csv(csv_path: str, experiments_matrix: List[List[ExperimentLoader]],
                                         labels: List[str], datasets: List[str], alpha=0.01):
    _save_default_hypothesis_csv(csv_path, experiments_matrix, labels, datasets, alpha, _get_avg_sqr_error_data)


# --------------------------------------------------------------------------------------------------------------------
# SPECIFIC TABLE GENERATION
# ---------------------------------------------------------------------------------------------------------------------

def _save_default_kruskall_hypothesis_text(text_path: str, experiments_matrix: List[List[ExperimentLoader]],
                                           labels: List[str], datasets: List[str], alpha: float,
                                           data_method: Callable[[List[ExperimentLoader]], List[np.array]]):

    assert len(experiments_matrix) == len(datasets), "first dimension of experiment matrix must have the same length as datasets"
    assert len(experiments_matrix[0]) == len(labels), "second dimension of experiment matrix must have the same length as labels"

    empty_str = ""
    for exp_list in experiments_matrix:

        data = data_method(exp_list)
        try:
            _, p = kruskal(*data)

            if p <= alpha:
                # reject the null hypothesis, are not the same
                empty_str += "\u2714"
            else:
                # cannot reject the null hypothesis
                empty_str += "\u2716"
        except ValueError:
            # if all the values are the same then don't reject the null hypothesis
            empty_str += "\u2592"

    print(empty_str)
    with open(text_path, 'w') as f:
        f.write(empty_str)

# --------------------------------------------------------------------------------------------------------------------
# SPECIFIC TABLE GENERATION
# ---------------------------------------------------------------------------------------------------------------------


def save_max_sum_hv_kruskall_hypothesis_table_csv(csv_path: str, experiments_matrix: List[List[ExperimentLoader]],
                                                  labels: List[str], datasets: List[str], alpha=0.01):
    _save_default_kruskall_hypothesis_text(csv_path, experiments_matrix, labels, datasets, alpha, _get_max_sum_hv_data)


def save_avg_sqr_error_kruskall_hypothesis_table_csv(csv_path: str, experiments_matrix: List[List[ExperimentLoader]],
                                                     labels: List[str], datasets: List[str], alpha=0.01):
    _save_default_kruskall_hypothesis_text(csv_path, experiments_matrix, labels, datasets, alpha, _get_avg_sqr_error_data)
