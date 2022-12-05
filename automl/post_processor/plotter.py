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
    fig.savefig(
        os.path.join(path, f"{mode}.png"),
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
    ax.bar_label(bar, list(df[all_series].idxmax(axis=1)))
    for i, series in enumerate(comparison):
        ax.bar(
            paddings[i + 1],
            df[series],
            width,
            label=series,
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
    fig.savefig(
        os.path.join(path, "comparison.png"),
        bbox_extra_artists=(lgd, text),
        bbox_inches="tight",
    )


def create_time_plot(df, others, ticks, paddings, width, labels, path):
    # df[
    #     [f"argumentation_time_{x}" for x in others if f"argumentation_time_{x}" in df]
    #     + [f"automl_time_{x}" for x in others if f"automl_time_{x}" in df]
    # ].plot.bar().get_figure().savefig(os.path.join(path, "time.png"))
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
        bbox_to_anchor=(0.5, -0.1),
    )
    text = fig.text(-0.2, 1.05, "", transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig(
        os.path.join(path, "time.png"),
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


def time_plot(summary, output_path, budget):
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
                    reward["absolute_time"]
                    - result["start_time"]
                    - (
                        result[f"""graph_generation_time_{reward["iteration"]}"""]
                        + result[f"""space_generation_time_{reward["iteration"]}"""]
                        if reward["iteration"] != 1
                        else 0
                    )
                    # if "absolute_time" in reward
                    # else np.nan
                    for reward in result["evaluated_rewards"]
                    if "absolute_time" in reward
                    and "balanced_accuracy" in reward
                    and reward["balanced_accuracy"] != "-inf"
                ]
                # timing = (
                #     pd.DataFrame(timing)
                #     .interpolate()
                #     .ffill()
                #     .bfill()
                #     .iloc[:, 0]
                #     .values.tolist()
                # )
                scores = [
                    max(
                        [
                            # 0 if _reward["balanced_accuracy"] == "-inf"
                            # or "absolute_time" not in _reward
                            # else
                            _reward["balanced_accuracy"]
                            for _reward in result["evaluated_rewards"][: (idx + 1)]
                            if "balanced_accuracy" in _reward
                            and _reward["balanced_accuracy"] != "-inf"
                        ]
                    )
                    for idx, reward in enumerate(result["evaluated_rewards"])
                    if "absolute_time" in reward
                    and "balanced_accuracy" in reward
                    and reward["balanced_accuracy"] != "-inf"
                ]
                iterations = [
                    reward["iteration"]
                    for reward in result["evaluated_rewards"]
                    if "absolute_time" in reward
                    and "balanced_accuracy" in reward
                    and reward["balanced_accuracy"] != "-inf"
                ]
                markers = [
                    iterations.index(iteration)
                    for iteration in range(1, max(iterations) + 1)
                    if iteration in iterations
                ]
                results[approach][dataset] = {
                    "index": current_subplot_index,
                    "title": dataset_names[current_subplot_index],
                    "timing": timing[1:],
                    "scores": scores[1:],
                    "markers": markers,
                }
                results[approach][dataset]["min_score"] = min(
                    results[approach][dataset]["scores"]
                )
                results[approach][dataset]["max_score"] = max(
                    results[approach][dataset]["scores"]
                )

    fig, axs = plt.subplots(1, 5)
    min_absolute_time = min(
        [
            min([min(results[approach][dataset]["timing"]) for dataset in dataset_ids])
            for approach in approaches
        ]
    )
    max_absolute_time = max(
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
    markers = {
        "baseline": "o",
        "pkb": "^",
        "ika": "s",
        "pkb_ika": "*",
    }
    colors = {
        "baseline": "tab:blue",
        "pkb": "tab:orange",
        "ika": "tab:green",
        "pkb_ika": "tab:red",
    }
    for approach in approaches:
        for dataset in dataset_ids:
            axs[results[approach][dataset]["index"]].title.set_text(
                results[approach][dataset]["title"]
            )
            timing = (
                [min_absolute_time]
                + results[approach][dataset]["timing"]
                # + [max_absolute_time]
            )
            timing = [time / 60 for time in timing]
            # timing = [time / 60 for time in results[approach][dataset]["timing"]]
            scores = (
                [results[approach][dataset]["min_score"]]
                + results[approach][dataset]["scores"]
                # + [results[approach][dataset]["max_score"]]
            )
            # scores = results[approach][dataset]["scores"]
            scores = [
                (score - min_absolute_score[dataset])
                / (max_absolute_score[dataset] - min_absolute_score[dataset])
                for score in scores
            ]
            axs[results[approach][dataset]["index"]].plot(
                timing,
                scores,
                label=approach,
                marker=markers[approach],
                markevery=len(timing) - 1,
                markersize=9 if approach == "pkb_ika" else 6,
                color=colors[approach],
            )
            for idx in results[approach][dataset]["markers"]:
                axs[results[approach][dataset]["index"]].plot(
                    timing[idx],
                    scores[idx],
                    marker=markers[approach],
                    markersize=9 if approach == "pkb_ika" else 6,
                    color=colors[approach],
                )
            axs[results[approach][dataset]["index"]].set_xlabel(
                "optimization time (m)", labelpad=10
            )
            axs[results[approach][dataset]["index"]].set_ylabel(
                "balanced accuracy", labelpad=7
            )
            # axs[results[approach][dataset]["index"]].set_ylim([0.0, 1])
            # axs[results[approach][dataset]["index"]].set_xlim(
            #     [0, 60 if budget == 500 else 120]
            # )
            axs[results[approach][dataset]["index"]].set_xlim(
                [0, 80 if budget == 500 else 120]
            )
            ticks = [
                round(tick, 3)
                for tick in np.linspace(0.0, 1.0, num=7)
                * (max_absolute_score[dataset] - min_absolute_score[dataset])
                + min_absolute_score[dataset]
            ]
            axs[results[approach][dataset]["index"]].set_yticklabels([ticks[0]] + ticks)
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
