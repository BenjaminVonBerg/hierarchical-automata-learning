import argparse
import itertools
import os
from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame

from evaluate.generate_results import RESULTS_FILE, ALGORITHM

config_cols = ['num_individual_mdps', 'num_states_per_individual_mdp', 'num_inputs', 'num_outputs',
               'num_transitions_per_connected_mdps', 'connection_type', 'num_traces', 'len_traces']

metric_cols = {
    # base column name : pretty name
    "bisimilarity_max": "Bisimilarity (Max)",
    "bisimilarity_mean": "Bisimilarity (Mean)",
    "compliance": "Compliance",
}


def get_full_metric_col_names(base_col_name: str, include_diff: bool = True):
    """ Get the full column names for the given base column name"""
    hal_col = "hal_" + base_col_name
    alg_col = "alg_" + base_col_name
    diff_col = "diff_" + base_col_name
    if include_diff:
        return hal_col, alg_col, diff_col
    else:
        return hal_col, alg_col


def get_full_pretty_name(full_col_name: str, metric=True, desc=True):
    prefix, base_col_name = full_col_name.split("_", 1)
    pretty_name = metric_cols.get(base_col_name, base_col_name)

    _a = {'hal': "HAL",
          'alg': ALGORITHM,
          'diff': f"HAL vs {ALGORITHM}"}

    if metric and desc:
        return f"{pretty_name} ({_a[prefix]})"
    elif metric:
        return f"{pretty_name}"
    elif desc:
        return f"{_a[prefix]}"


def get_metric_from_col_name(col_name: str):
    for base_col_name in metric_cols:
        if base_col_name in col_name:
            return metric_cols[base_col_name]


def get_df(file=RESULTS_FILE):
    """ Load dataframe and add the diff_ columns for the metrics"""
    print(f"reading from {file}")
    df = pd.read_csv(file)
    for col_name in metric_cols:
        hal_col, alg_col, diff_col = get_full_metric_col_names(col_name)
        df[diff_col] = df[hal_col] - df[alg_col]
    return df


def print_metric_col_descriptions(df: DataFrame):
    """ Print descriptions of the metric columns """
    for col_name, pretty_name in metric_cols.items():
        hal_col, alg_col, diff_col = get_full_metric_col_names(col_name)
        hal_desc = df[hal_col].describe()
        alg_desc = df[alg_col].describe()
        diff_desc = df[diff_col].describe()

        print(pretty_name)
        print(pd.concat([hal_desc, alg_desc, diff_desc],
                        keys=['HAL', 'ALG', "Diff"],
                        axis=1))
        print()

def print_simple_metric_analysis(df : DataFrame):
    percentage_key = "Worse Cases (%)"
    improvement_key = "Avg. Impr."
    deterioration_key = "Avg. Deter."

    stat_df = pd.DataFrame(columns=[percentage_key, improvement_key, deterioration_key])
    for col_name, pretty_name in metric_cols.items():
        idx = "diff_" + col_name
        worse_indices = df[idx] < 0
        stat_dict = {
            improvement_key: np.average(df[idx][np.logical_not(worse_indices)]),
            deterioration_key: np.average(df[idx][(worse_indices)]),
            percentage_key: 100 *np.sum(worse_indices) / len(worse_indices),
        }
        for key, val in stat_dict.items():
            stat_df.loc[pretty_name, key] = val
    print(stat_df.to_markdown())
    print()
    print(stat_df.to_latex())

def show_plot(df: DataFrame, plot_type: str, columns: Sequence[str], title: str = None, label_method: str = None, x_label: str = None, y_label: str = None):
    plt.close()

    if label_method is None or label_method == "full":
        labels = [get_full_pretty_name(c) for c in columns]
    elif label_method == "metric":
        labels = [get_full_pretty_name(c, desc=False) for c in columns]
    elif label_method == "desc":
        labels = [get_full_pretty_name(c, metric=False) for c in columns]

    if title is None:
        common_metric = get_metric_from_col_name(columns[0])
        if all(get_metric_from_col_name(c) == common_metric for c in columns):
            title = common_metric
        else:
            common_metric = title = None

    if all(l.startswith(title) for l in labels):
        labels = [l[len(title):].replace("(", "").replace(")", "") for l in labels]

    match plot_type:
        case "violin":
            fig, ax = plt.subplots()
            ax.violinplot(dataset=[df[c] for c in columns])
            ax.set_xlim(0.25, len(labels) + 0.75)
            ax.set_xticks(np.arange(1, len(labels) + 1), labels=labels)
        case "box":
            ax = df.boxplot(column=list(columns), vert=False)
            ax.set_yticklabels(labels)
        case _:
            raise NotImplementedError

    if x_label: ax.set_xlabel(x_label)
    if y_label: ax.set_ylabel(y_label)

    ax.set_title(title)
    #plt.ylim((-0.05, 1.05))
    plt.tight_layout()
    plt.show()

def plot_metric(df: DataFrame, plot_type: str, metric: str, include_diff: bool = False):
    title = metric_cols[metric]
    show_plot(df, plot_type, get_full_metric_col_names(metric, include_diff=include_diff), title=title)


def plot_diffs(df: DataFrame, plot_type: str):
    title = f"Differences in Metrics"
    show_plot(df, plot_type, [f"diff_{metric}" for metric in metric_cols], title=title, label_method="metric", x_label="m(M_HAL) - m(M_IOA)")

def plot_cols_for_params(df: DataFrame, params: list[str], metrics: list[str]):
    df.boxplot(column=metrics, by=params, vert=False)
    plt.tight_layout()
    plt.show()
    #plt.savefig('Test', bbox_inches='tight')

    ranges = {param : sorted(set(df[param])) for param in params}
    combinations = list(itertools.product(*ranges.values()))
    for combination in combinations:
        key_dict = dict(zip(params, combination))
        print(key_dict)
        selection = df[np.logical_and(True, *(df[param] == value for param, value in key_dict.items()))]
        selection = selection[metrics]
        print(selection.describe())
        print()


def analyze_results(file: str = RESULTS_FILE):
    df = get_df(file)
    print_metric_col_descriptions(df)
    print_simple_metric_analysis(df)

    params = [
        ["num_states_per_individual_mdp","connection_type"],
        ["num_states_per_individual_mdp"],
        ["connection_type"],
        ["num_individual_mdps"],
        ["num_inputs"],
        ["num_outputs"],
        ["num_transitions_per_connected_mdps"],
        ["num_traces"],
        ["len_traces"],
    ]

    metrics = [
        # "diff_bisimilarity_max",
        # "diff_bisimilarity_mean",
        "alg_bisimilarity_mean",
        "hal_bisimilarity_mean",
        # "diff_compliance",
    ]

    for param in params:
        plot_cols_for_params(df, param, metrics)

    cols = [x for metric in ["bisimilarity_max", "bisimilarity_mean", "compliance"] for x in get_full_metric_col_names(metric, False)]
    show_plot(df, "box", cols, "Results of Metrics", x_label="m(M)")

    plot_diffs(df, "box")

    for metric in metric_cols:
        plot_metric(df, "box", metric)
