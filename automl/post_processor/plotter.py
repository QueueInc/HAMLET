from importlib.resources import path
import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from paretoset import paretoset
import copy


def plot_pd(df, baseline, others, path):
    f = plt.figure()
    df.boxplot([f"delta_{x}" for x in others])
    for ext in ["png", "pdf"]:
        f.savefig(os.path.join(path, f"{baseline}_boxplot_delta.{ext}"))

    f = plt.figure()
    df.boxplot([f"delta_iteration_{x}" for x in others])
    for ext in ["png", "pdf"]:
        f.savefig(os.path.join(path, f"{baseline}_boxplot_delta_iteration.{ext}"))

    for ext in ["png", "pdf"]:
        df[[f"delta_iteration_{x}" for x in others]].plot.bar().get_figure().savefig(
            os.path.join(path, f"{baseline}_barchart_delta_iteration.{ext}")
        )

    for ext in ["png", "pdf"]:
        df[[f"norm_iteration_{x}" for x in others]].plot.bar().get_figure().savefig(
            os.path.join(path, f"{baseline}_barchart_norm_iteration.{ext}")
        )

    for ext in ["png", "pdf"]:
        df[[f"delta_{x}" for x in others]].plot.bar().get_figure().savefig(
            os.path.join(path, f"{baseline}_barchart_delta.{ext}")
        )

    for ext in ["png", "pdf"]:
        df[[f"norm_{x}" for x in others]].plot.bar().get_figure().savefig(
            os.path.join(path, f"{baseline}_barchart_norm_delta.{ext}")
        )


def create_hamlet_plot(
    df, all_series, ticks, paddings, width, labels, mode, support_map, path
):
    fig, ax = plt.subplots()
    for i, series in enumerate(all_series):
        label = series if ("_" not in series) else " + ".join(series.split("_"))
        label = label if label == "baseline" else label.upper()
        ax.bar(
            paddings[i],
            df[support_map["prefix"][mode] + series],
            width,
            label=label,
        )
    ax.set_ylabel(
        support_map["y_label"][mode],
        labelpad=10,
    )

    # ax.set_title("Balanced accuracy achieved by the approaches")
    ax.set_xticks(ticks, labels)
    if mode == "accuracy":
        ax.set_ylim([0.75, 1.01])
    # ax.legend()

    _handles, _labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(_labels, _handles))
    lgd = fig.legend(
        by_label.values(),
        by_label.keys(),
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, -0.1),
    )
    text = fig.text(-0.2, 1.05, "", transform=ax.transAxes)
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(
            os.path.join(path, f"{mode}.{ext}"),
            bbox_extra_artists=(lgd, text),
            bbox_inches="tight",
        )


def create_comparison_plot(
    df, all_series, comparison, ticks, paddings, width, labels, path
):
    fig, ax = plt.subplots()
    bar = ax.bar(
        paddings[0],
        list(df[all_series].max(axis=1)),
        width,
        label="HAMLET",
        color="tab:cyan",
    )
    # ax.bar_label(bar, list(df[all_series].idxmax(axis=1)))
    for i, series in enumerate(comparison):
        ax.bar(
            paddings[i + 1],
            df[series],
            width,
            label="Auto-sklearn" if series == "auto_sklearn" else "H2O",
            color="tab:pink" if series == "auto_sklearn" else "tab:brown",
        )
    ax.set_ylabel(
        "Balanced accuracy",
        labelpad=10,
    )

    # ax.set_title("Balanced accuracy achieved by the approaches")
    ax.set_xticks(ticks, labels)
    ax.set_ylim([0.75, 1.01])
    # ax.legend()

    _handles, _labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(_labels, _handles))
    lgd = fig.legend(
        by_label.values(),
        by_label.keys(),
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.1),
    )
    text = fig.text(-0.2, 1.05, "", transform=ax.transAxes)
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(
            os.path.join(path, f"comparison.{ext}"),
            bbox_extra_artists=(lgd, text),
            bbox_inches="tight",
        )


def create_time_plot(df, others, ticks, paddings, width, labels, path):
    # df[
    #     [f"argumentation_time_{x}" for x in others if f"argumentation_time_{x}" in df]
    #     + [f"automl_time_{x}" for x in others if f"automl_time_{x}" in df]
    # ].plot.bar().get_figure().savefig(os.path.join(path, "time.pdf"))
    colors = {
        "pkb": "tab:orange",
        "ika": "tab:green",
        "pkb_ika": "tab:red",
    }
    fig, ax = plt.subplots()
    for i, series in enumerate(others):
        label = series if ("_" not in series) else " + ".join(series.split("_"))
        label = label if label == "baseline" else label.upper()
        total = df[[f"argumentation_time_{series}", f"automl_time_{series}"]].sum(
            axis=1
        )
        argumentation = df[f"argumentation_time_{series}"] / total
        automl = df[f"automl_time_{series}"] / total
        ax.bar(
            paddings[i],
            automl,
            width,
            label="automl " + label,
            color=colors[series],
            edgecolor="black",
            linewidth=1,
            # hatch="oo",
        )
        ax.bar(
            paddings[i],
            argumentation,
            width,
            label="argum. " + label,
            color=colors[series],
            bottom=automl,
            hatch="xx",
            edgecolor="black",
            linewidth=1,
        )
    ax.set_ylabel(
        "Percentage of time",
        labelpad=10,
    )

    # ax.set_title("Balanced accuracy achieved by the approaches")
    ax.set_xticks(ticks, labels)
    # ax.legend()

    _handles, _labels = plt.gca().get_legend_handles_labels()
    by_label = dict(
        zip(
            _labels[:2][::-1] + _labels[2:4][::-1] + _labels[4:][::-1],
            _handles[:2][::-1] + _handles[2:4][::-1] + _handles[4:][::-1],
        )
    )
    lgd = fig.legend(
        by_label.values(),
        by_label.keys(),
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.13),
    )
    text = fig.text(-0.2, 1.05, "", transform=ax.transAxes)
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(
            os.path.join(path, f"time.{ext}"),
            bbox_extra_artists=(lgd, text),
            bbox_inches="tight",
        )


def plot_matplotlib(df, baseline, others, comparison, path):

    SMALL_SIZE = 14
    MEDIUM_SIZE = 14

    plt.rc("font", size=MEDIUM_SIZE)  # controls default text sizes
    plt.rc("axes", titlesize=MEDIUM_SIZE)  # fontsize of the axes title
    plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("legend", fontsize=MEDIUM_SIZE)  # legend fontsize
    plt.rc("figure", titlesize=MEDIUM_SIZE)  # fontsize of the figure title

    labels = df["name"]
    all_series = [baseline] + others
    ticks = np.arange(len(labels))
    width = 0.2
    paddings_5 = [
        ticks - (width / 2 * 3),
        ticks - (width / 2),
        ticks + (width / 2),
        ticks + (width / 2 * 3),
    ]
    paddings_3 = [
        ticks - width,
        ticks,
        ticks + width,
    ]
    support_map = {
        "prefix": {
            "accuracy": "",
            "iterations": "iteration_",
            "best_time": "best_time_",
        },
        "y_label": {
            "accuracy": "Balanced accuracy",
            "iterations": "#explored pipeline instances",
            "best_time": "#optimization time (m)",
        },
    }

    create_hamlet_plot(
        df, all_series, ticks, paddings_5, width, labels, "accuracy", support_map, path
    )
    create_hamlet_plot(
        df,
        all_series,
        ticks,
        paddings_5,
        width,
        labels,
        "iterations",
        support_map,
        path,
    )
    create_hamlet_plot(
        df, all_series, ticks, paddings_5, width, labels, "best_time", support_map, path
    )
    create_comparison_plot(
        df, all_series, comparison, ticks, paddings_3, width, labels, path
    )

    create_time_plot(df, others, ticks, paddings_3, width, labels, path)


def time_plot(summary, output_path, budget, mode):
    def create_figure():
        # fig = plt.figure(figsize=(15, 7), layout="constrained")
        # spec = fig.add_gridspec(2, 6)
        # axs = []
        # axs.append(fig.add_subplot(spec[0, 1:3]))
        # axs.append(fig.add_subplot(spec[0, 3:5]))
        # axs.append(fig.add_subplot(spec[1, :2]))
        # axs.append(fig.add_subplot(spec[1, 2:4]))
        # axs.append(fig.add_subplot(spec[1, 4:]))
        fig, axs = plt.subplots(1, 5)
        fig.set_size_inches(30, 5)
        return fig, axs

    def create_time_iteration_plot(results):
        fig, axs = create_figure()
        colors = {
            "baseline": "tab:blue",
            "pkb": "tab:orange",
            "ika": "tab:green",
            "pkb_ika": "tab:red",
        }
        thickness = {
            "baseline": 4,
            "pkb": 3,
            "ika": 2,
            "pkb_ika": 1,
        }
        for approach in approaches:
            for dataset in dataset_ids:
                axs[results[approach][dataset]["index"]].title.set_text(
                    results[approach][dataset]["title"]
                )
                timing = [0.0] + results[approach][dataset]["timing"]
                if mode == "time":
                    timing = [time / 60 for time in timing]
                results[approach][dataset]["scores"] = [
                    score
                    if score != float("-inf")
                    else min(
                        [
                            elem
                            for _approach in (
                                ["pkb", "pkb_ika"]
                                if approach in ["pkb", "pkb_ika"]
                                else ["baseline", "ika"]
                            )
                            for elem in results[_approach][dataset]["scores"]
                            if elem != float("-inf")
                        ]
                    )
                    for score in results[approach][dataset]["scores"]
                ]
                scores = [
                    max(
                        results[approach][dataset]["min_score"],
                        results[approach][dataset]["scores"][0],
                    )
                ] + results[approach][dataset]["scores"]
                scores = [
                    (score - min_absolute_score[dataset])
                    / (max_absolute_score[dataset] - min_absolute_score[dataset])
                    for score in scores
                ]
                label = (
                    approach
                    if ("_" not in approach)
                    else " + ".join(approach.split("_"))
                )
                label = label if label == "baseline" else label.upper()
                # scores = [
                #     score for idx, score in enumerate(scores) if timing[idx] <= 60
                # ]
                # timing = [time for time in timing if time <= 60]
                axs[results[approach][dataset]["index"]].plot(
                    timing,
                    scores,
                    label=label,
                    # marker=markers[approach],
                    # markevery=len(timing) - 1,
                    # markersize=9 if approach == "pkb_ika" else 6,
                    color=colors[approach],
                    linewidth=thickness[approach],
                )
                # for idx in results[approach][dataset]["markers"]:
                #     if idx != 0:
                #         axs[results[approach][dataset]["index"]].plot(
                #             timing[idx],
                #             scores[idx],
                #             marker=markers[approach],
                #             markersize=9 if approach == "pkb_ika" else 6,
                #             color=colors[approach],
                #         )
                axs[results[approach][dataset]["index"]].set_xlabel(
                    "Optimization time (m)", labelpad=10
                )
                if results[approach][dataset]["index"] == 0:
                    axs[results[approach][dataset]["index"]].set_ylabel(
                        "Balanced accuracy", labelpad=7
                    )
                if mode == "time":
                    axs[results[approach][dataset]["index"]].set_xlim(
                        [-5, 60 if budget == 500 else 120]
                    )
                ticks = [
                    round(tick, 3)
                    for tick in np.linspace(0.0, 1.0, num=6)
                    * (max_absolute_score[dataset] - min_absolute_score[dataset])
                    + min_absolute_score[dataset]
                ]
                axs[results[approach][dataset]["index"]].set_yticklabels(
                    [ticks[0]] + ticks
                )
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        lgd = fig.legend(
            by_label.values(),
            by_label.keys(),
            loc="upper center",
            ncol=4,
            bbox_to_anchor=(0.5, -0.05),
        )
        text = fig.text(-0.2, 1.05, "", transform=axs[0].transAxes)
        for ext in ["png", "pdf"]:
            fig.savefig(
                os.path.join(output_path, f"accuracy_{mode}.{ext}"),
                bbox_extra_artists=(lgd, text),
                bbox_inches="tight",
            )

    def create_skyline(results):
        fig, axs = create_figure()
        scores = {}
        for dataset in dataset_ids:
            scores = [
                max([score for score in results[approach][dataset]["scores"]])
                for approach in approaches
            ]
            # timing = [
            #     results[approach][dataset]["timing"][
            #         results[approach][dataset]["scores"].index(scores[idx])
            #     ]
            #     for idx, approach in enumerate(approaches)
            # ]
            timing = [
                max([time for time in results[approach][dataset]["timing"]])
                for approach in approaches
            ]
            timing = [time / 60 for time in timing]
            timing = [time if time < 60 else 60 for time in timing]
            max_score, min_score = max(scores), min(scores)
            max_timing, min_timing = max(timing), min(timing)
            scores = [
                (score - min(scores)) / (max(scores) - min(scores)) for score in scores
            ]
            timing = [
                (time - min(timing)) / (max(timing) - min(timing)) for time in timing
            ]

            df = pd.DataFrame(
                {
                    "timing": timing,
                    "scores": scores,
                }
            )
            mask = paretoset(df, sense=["min", "max"])
            pareto = df[mask]
            colors = {
                "baseline": "tab:blue",
                "pkb": "tab:orange",
                "ika": "tab:green",
                "pkb_ika": "tab:red",
            }
            pareto = [tuple(x) for x in pareto.to_numpy()]
            # axs[results[approaches[0]][dataset]["index"]].scatter(
            #     pareto["timing"], pareto["scores"], color="orange", s=250, alpha=0.7
            # )
            for idx, approach in enumerate(approaches):
                label = (
                    approach
                    if ("_" not in approach)
                    else " + ".join(approach.split("_"))
                )
                label = label if label == "baseline" else label.upper()
                axs[results[approach][dataset]["index"]].title.set_text(
                    results[approach][dataset]["title"]
                )
                # axs[results[approach][dataset]["index"]].plot(
                #     np.linspace(0, 1, num=100),
                #     np.sqrt(1 - np.linspace(-1.0, 0, num=100) ** 2),
                #     color="black",
                # )
                axs[results[approach][dataset]["index"]].scatter(
                    timing[idx],
                    scores[idx],
                    color=colors[approach],
                    label=label,
                    s=250 if (timing[idx], scores[idx]) in pareto else 100,
                )
                axs[results[approach][dataset]["index"]].set_xlabel(
                    "Optimization time (m)", labelpad=10
                )
                if results[approach][dataset]["index"] == 0:
                    axs[results[approach][dataset]["index"]].set_ylabel(
                        "Balanced accuracy", labelpad=7
                    )
                # axs[results[approach][dataset]["index"]].set_xlim(
                #     [-5, 60 if budget == 500 else 120]
                # )
                ticks = [
                    round(tick, 3)
                    for tick in np.linspace(0.0, 1.0, num=7) * (max_score - min_score)
                    + min_score
                ]
                axs[results[approach][dataset]["index"]].set_yticklabels(
                    [ticks[0]] + ticks
                )
                ticks = [
                    round(tick, 1)
                    for tick in np.linspace(0.0, 1.0, num=6) * (max_timing - min_timing)
                    + min_timing
                ]
                axs[results[approach][dataset]["index"]].set_xticklabels(
                    [ticks[0]] + ticks
                )

        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        lgd = fig.legend(
            by_label.values(),
            by_label.keys(),
            loc="upper center",
            ncol=4,
            bbox_to_anchor=(0.5, -0.05),
        )
        text = fig.text(-0.2, 1.05, "", transform=axs[0].transAxes)
        fig.set_size_inches(30, 5)
        for ext in ["pdf", "png"]:
            fig.savefig(
                os.path.join(output_path, f"skyline.{ext}"),
                bbox_extra_artists=(lgd, text),
                bbox_inches="tight",
            )

    approaches = ["baseline", "pkb", "ika", "pkb_ika"]
    dataset_names = list(summary["name"])
    dataset_ids = list(summary.index.astype(str))

    SMALL_SIZE = 17
    MEDIUM_SIZE = 20

    plt.rc("font", size=MEDIUM_SIZE)  # controls default text sizes
    plt.rc("axes", titlesize=MEDIUM_SIZE)  # fontsize of the axes title
    plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("legend", fontsize=MEDIUM_SIZE)  # legend fontsize
    plt.rc("figure", titlesize=MEDIUM_SIZE)  # fontsize of the figure title

    results = {}
    for approach in approaches:
        results[approach] = {}
        with open(os.path.join(output_path, approach, "summary.json")) as f:
            summary = json.load(f)
            for dataset, result in summary.items():
                current_subplot_index = dataset_ids.index(dataset)
                timing = [
                    reward["total_time"] if mode == "time" else idx
                    for idx, reward in enumerate(result["evaluated_rewards"])
                    if "balanced_accuracy" in reward
                ]
                if mode == "time":
                    timing = [
                        sum(timing[: (len(timing) - idx)])
                        for idx, _ in enumerate(timing[::-1])
                    ]
                    timing = timing[::-1]
                scores = [
                    max(
                        [
                            float(_reward["balanced_accuracy"])
                            for _reward in result["evaluated_rewards"][: (idx + 1)]
                            if "balanced_accuracy" in _reward
                        ]
                    )
                    for idx, reward in enumerate(result["evaluated_rewards"])
                    if "balanced_accuracy" in reward
                ]
                iterations = [
                    reward["iteration"]
                    for reward in result["evaluated_rewards"]
                    if "balanced_accuracy" in reward
                ]
                markers = [
                    iterations.index(iteration)
                    for iteration in range(1, max(iterations) + 1)
                    if iteration in iterations
                ]
                results[approach][dataset] = {
                    "index": current_subplot_index,
                    "title": dataset_names[current_subplot_index],
                    "timing": timing,
                    "scores": scores,
                    "markers": markers,
                }
                results[approach][dataset]["min_score"] = min(
                    results[approach][dataset]["scores"]
                )
                results[approach][dataset]["max_score"] = max(
                    results[approach][dataset]["scores"]
                )

    max_absolute_score = {
        dataset: max(
            [max(results[approach][dataset]["scores"]) for approach in approaches]
        )
        for dataset in dataset_ids
    }
    min_absolute_score = {
        dataset: min(
            [
                min(
                    [
                        score
                        for score in results[approach][dataset]["scores"]
                        if score != float("-inf")
                    ]
                )
                for approach in approaches
            ]
        )
        for dataset in dataset_ids
    }

    create_time_iteration_plot(copy.deepcopy(results))
    if mode == "time":
        create_skyline(copy.deepcopy(results))
