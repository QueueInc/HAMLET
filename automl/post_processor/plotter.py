from importlib.resources import path
import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_pd(df, baseline, others, path):
    f = plt.figure()
    df.boxplot([f"delta_{x}" for x in others])
    f.savefig(os.path.join(path, f"{baseline}_boxplot_delta.png"))

    f = plt.figure()
    df.boxplot([f"delta_iteration_{x}" for x in others])
    f.savefig(os.path.join(path, f"{baseline}_boxplot_delta_iteration.png"))

    df[[f"delta_iteration_{x}" for x in others]].plot.bar().get_figure().savefig(
        os.path.join(path, f"{baseline}_barchart_delta_iteration.png")
    )

    df[[f"norm_iteration_{x}" for x in others]].plot.bar().get_figure().savefig(
        os.path.join(path, f"{baseline}_barchart_norm_iteration.png")
    )

    df[[f"delta_{x}" for x in others]].plot.bar().get_figure().savefig(
        os.path.join(path, f"{baseline}_barchart_delta.png")
    )

    df[[f"norm_{x}" for x in others]].plot.bar().get_figure().savefig(
        os.path.join(path, f"{baseline}_barchart_norm_delta.png")
    )


def plot_matplotlib(df, baseline, others, path):

    SMALL_SIZE = 14
    MEDIUM_SIZE = 14
    BIGGER_SIZE = 16

    plt.rc("font", size=MEDIUM_SIZE)  # controls default text sizes
    plt.rc("axes", titlesize=MEDIUM_SIZE)  # fontsize of the axes title
    plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("legend", fontsize=MEDIUM_SIZE)  # legend fontsize
    plt.rc("figure", titlesize=MEDIUM_SIZE)  # fontsize of the figure title

    df[
        [f"argumentation_time_{x}" for x in others if f"argumentation_time_{x}" in df]
        + [f"automl_time_{x}" for x in others if f"automl_time_{x}" in df]
    ].plot.bar().get_figure().savefig(os.path.join(path, f"time.png"))

    labels = df["name"]
    x = np.arange(len(labels))  # the label locations
    width = 0.1  # the width of the bars

    fig, ax = plt.subplots()
    all_series = [baseline] + others
    paddings = [
        x - (width / 2 * 5),
        x - (width / 2 * 3),
        x - (width / 2),
        x + (width / 2),
        x + (width / 2 * 3),
        x + (width / 2 * 5),
    ]
    for i in range(len(all_series)):
        label = (
            all_series[i]
            if "_" in all_series[i]
            else " + ".join(all_series[i].split("_"))
        )
        ax.bar(paddings[i], df[all_series[i]], width, label=label)
    ax.set_ylabel("Balanced accuracy", labelpad=10)
    # ax.set_title("Balanced accuracy achieved by the approaches")
    ax.set_xticks(x, labels)
    ax.set_ylim([0.75, 1])
    # ax.legend()

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    lgd = fig.legend(
        by_label.values(),
        by_label.keys(),
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, -0.1),
    )
    text = fig.text(-0.2, 1.05, "", transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig(
        os.path.join(path, f"accuracy.png"),
        bbox_extra_artists=(lgd, text),
        bbox_inches="tight",
    )

    labels = df["name"]
    x = np.arange(len(labels))  # the label locations
    width = 0.1  # the width of the bars
    fig, ax = plt.subplots()
    for i in range(len(all_series)):
        label = (
            all_series[i]
            if "_" in all_series[i]
            else " + ".join(all_series[i].split("_"))
        ).upper()
        ax.bar(paddings[i], df[f"iteration_{all_series[i]}"], width, label=label)
    ax.set_ylabel("#explored pipeline instances", labelpad=10)
    # ax.set_title("No. configuration in which the approaches achieve the best result")
    ax.set_xticks(x, labels)
    # ax.legend()

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    lgd = fig.legend(
        by_label.values(),
        by_label.keys(),
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, -0.1),
    )
    text = fig.text(-0.2, 1.05, "", transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig(
        os.path.join(path, f"iterations.png"),
        bbox_extra_artists=(lgd, text),
        bbox_inches="tight",
    )

    labels = df["name"]
    x = np.arange(len(labels))  # the label locations
    width = 0.1  # the width of the bars
    fig, ax = plt.subplots()
    for i in range(len(all_series)):
        label = (
            all_series[i]
            if "_" in all_series[i]
            else " + ".join(all_series[i].split("_"))
        ).upper()
        ax.bar(paddings[i], df[f"best_time_{all_series[i]}"], width, label=label)
    ax.set_ylabel("#optimization time (m)", labelpad=10)
    # ax.set_title("No. configuration in which the approaches achieve the best result")
    ax.set_xticks(x, labels)
    # ax.legend()

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    lgd = fig.legend(
        by_label.values(),
        by_label.keys(),
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, -0.1),
    )
    text = fig.text(-0.2, 1.05, "", transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig(
        os.path.join(path, f"best_time.png"),
        bbox_extra_artists=(lgd, text),
        bbox_inches="tight",
    )


def time_plot(summary, output_path):
    approaches = ["baseline", "pkb", "ika", "pkb_ika"]
    dataset_names = list(summary["name"])
    dataset_ids = list(summary.index.astype(str))
    # num_subplots = len(next(os.walk(os.path.join(output_path, approaches[0])))[1])

    results = {}
    for approach in approaches:
        results[approach] = {}
        with open(os.path.join(output_path, approach, "summary.json")) as f:
            summary = json.load(f)
            for dataset, result in summary.items():
                current_subplot_index = dataset_ids.index(dataset)
                timing = [
                    reward["absolute_time"] - result["start_time"]
                    for reward in result["evaluated_rewards"]
                    if "absolute_time" in reward
                ]
                scores = [
                    max(
                        [
                            float(_reward["balanced_accuracy"])
                            for _reward in result["evaluated_rewards"][: (idx + 1)]
                            if "balanced_accuracy" in _reward
                        ]
                    )
                    for idx, reward in enumerate(result["evaluated_rewards"])
                    if "absolute_time" in reward
                ]
                results[approach][dataset] = {
                    "index": current_subplot_index,
                    "title": dataset_names[current_subplot_index],
                    "timing": timing[1:],
                    "scores": scores[1:],
                }
                results[approach][dataset]["min_score"] = min(
                    results[approach][dataset]["scores"]
                )
                results[approach][dataset]["max_score"] = max(
                    results[approach][dataset]["scores"]
                )

    fig, axs = plt.subplots(1, 5)
    min_time = min(
        [
            min([min(results[approach][dataset]["timing"]) for dataset in dataset_ids])
            for approach in approaches
        ]
    )
    max_time = max(
        [
            max([max(results[approach][dataset]["timing"]) for dataset in dataset_ids])
            for approach in approaches
        ]
    )
    max_absolute_score = {
        dataset: max(
            [max(results[approach][dataset]["scores"]) for approach in approaches]
        )
        for dataset in dataset_ids
    }
    min_absolute_score = {
        dataset: min(
            [min(results[approach][dataset]["scores"]) for approach in approaches]
        )
        for dataset in dataset_ids
    }
    for approach in approaches:
        for dataset in dataset_ids:
            axs[results[approach][dataset]["index"]].title.set_text(
                results[approach][dataset]["title"]
            )
            timing = [min_time] + results[approach][dataset]["timing"] + [max_time]
            scores = (
                [results[approach][dataset]["min_score"]]
                + results[approach][dataset]["scores"]
                + [results[approach][dataset]["max_score"]]
            )
            scores = [
                (score - min_absolute_score[dataset])
                / (max_absolute_score[dataset] - min_absolute_score[dataset])
                for score in scores
            ]
            axs[results[approach][dataset]["index"]].plot(
                timing,
                scores,
                label=approach,
            )
            axs[results[approach][dataset]["index"]].set_xlabel(
                "optimization time (s)", labelpad=10
            )
            axs[results[approach][dataset]["index"]].set_ylabel(
                "balanced accuracy", labelpad=7
            )
            # axs[results[approach][dataset]["index"]].set_ylim([scores[5], 1])
            axs[results[approach][dataset]["index"]].set_xlim([50, max_time])
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    lgd = fig.legend(
        by_label.values(),
        by_label.keys(),
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.5, -0.1),
    )
    text = fig.text(-0.2, 1.05, "", transform=axs[0].transAxes)
    # fig.tight_layout()
    fig.set_size_inches(35, 5)
    fig.savefig(
        os.path.join(output_path, "accuracy_time.png"),
        bbox_extra_artists=(lgd, text),
        bbox_inches="tight",
    )
