import math
from typing import List
from logging import warning

import igraph
import matplotlib.pyplot as plt
import numpy as np
import seaborn
from scipy import stats
from sklearn.preprocessing import normalize

import sources.experiment.metrics as metrics
import sources.mongo_connection.mongo_queries as dbq
from sources.experiment.experiment_loader import ExperimentLoader
from sources.gas.auxiliary_funtions import average_odf_and_community_score
from sources.gloaders.loader_interface import LoaderInterface


def compare_times_taken(img_path: str, experiments_matrix: List[List[ExperimentLoader]], labels: List[str],
                        datasets: List[str], remove_outliers=False,
                        yticks=[1*60*60, 5*60*60, 12*60*60, 1*24*60*60, 2*24*60*60, 3*24*60*60, 4*24*60*60, 5*24*60*60],
                        ylabels=["1hour", "5hours", "12hours", "1day", "2days", "3days", "4days", "5days"],
                        item_w=2):

    font = {'family': 'DejaVu Sans',
            'weight': 'bold',
            'size': 30}

    f, axs = plt.subplots(1)
    f.set_size_inches(w=item_w*len(datasets), h=10)

    for i, experiments_list in enumerate(experiments_matrix):
        exp_times_list = [exp.get_times_matrix() for exp in experiments_list]

        if remove_outliers:
            exp_times_list = [metrics.remove_outliers(exp) for exp in exp_times_list]

        mean = np.array([np.mean(exp_times) for exp_times in exp_times_list])
        error = np.array([np.std(exp_times) for exp_times in exp_times_list])

        axs.errorbar(range(len(datasets)), mean, yerr=error, fmt='-o', label=labels[i])

    axs.set_ylabel("time taken (s)", fontdict=font)
    # axs.grid(color='gray', linestyle='-', linewidth=0.5, axis='y')

    axs.set_xticks(np.arange(len(datasets)))
    axs.set_yticks(yticks)
    xtickNames = plt.setp(axs, xticklabels=datasets, yticklabels=ylabels)
    plt.setp(xtickNames, rotation=45, fontsize=26)

    # Place a legend to the right of this smaller subplot.
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    plt.savefig(img_path)


def generations_total_comparison(img_path: str, experiments_matrix: List[List[ExperimentLoader]], labels: List[str],
                                 datasets: List[str], legend=True, item_w=2):

    font_big = {'family': 'DejaVu Sans',
                  'weight': 'bold',
                  'size': 30}

    f, axs = plt.subplots(1)
    f.set_size_inches(w=max(10, item_w * len(datasets)), h=10)

    for i, experiments_list in enumerate(experiments_matrix):
        generations_list = [exp.get_generations_taken_matrix() for exp in experiments_list]

        generations_sum = [np.sum(gen, axis=1) for gen in generations_list]

        mean = np.array([np.mean(exp_error) for exp_error in generations_sum])
        error = np.array([np.std(exp_error) for exp_error in generations_sum])

        axs.errorbar(range(len(datasets)), mean, yerr=error, fmt='-o', label=labels[i])

    axs.set_ylabel("median generations taken", fontdict=font_big)
    # axs.grid(color='gray', linestyle='-', linewidth=0.5, axis='y')

    axs.set_xticks(np.arange(len(datasets)))
    xtickNames = plt.setp(axs, xticklabels=datasets)
    plt.setp(xtickNames, rotation=45, fontsize=26)
    plt.setp(axs.get_yticklabels(), fontsize=26)

    # Place a legend to the right of this smaller subplot.
    if legend:
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    plt.savefig(img_path)
