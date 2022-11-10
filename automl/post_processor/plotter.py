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
        [f"argumentation_time_{x}" for x in others]
        + [f"automl_time_{x}" for x in others]
    ].plot.bar().get_figure().savefig(os.path.join(path, f"time.png"))

    labels = df["name"]
    x = np.arange(len(labels))  # the label locations
    width = 0.2  # the width of the bars

    fig, ax = plt.subplots()
    all_series = [baseline] + others
    paddings = [
        x - (width / 2 * 3),
        x - (width / 2),
        x + (width / 2),
        x + (width / 2 * 3),
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
    ax.set_ylim([0.6, 1])
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
    width = 0.2  # the width of the bars
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
