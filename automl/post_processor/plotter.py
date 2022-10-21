from importlib.resources import path
import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_pd(df, mode, other, path):
    f = plt.figure()
    df.boxplot([f"delta_{x}" for x in other])
    f.savefig(os.path.join(path, f"{mode}_boxplot_delta.png"))

    f = plt.figure()
    df.boxplot([f"delta_iteration_{x}" for x in other])
    f.savefig(os.path.join(path, f"{mode}_boxplot_delta_iteration.png"))

    df[[f"delta_iteration_{x}" for x in other]].plot.bar().get_figure().savefig(
        os.path.join(path, f"{mode}_barchart_delta_iteration.png")
    )

    df[[f"norm_iteration_{x}" for x in other]].plot.bar().get_figure().savefig(
        os.path.join(path, f"{mode}_barchart_norm_iteration.png")
    )

    df[[f"delta_{x}" for x in other]].plot.bar().get_figure().savefig(
        os.path.join(path, f"{mode}_barchart_delta.png")
    )

    df[[f"norm_{x}" for x in other]].plot.bar().get_figure().savefig(
        os.path.join(path, f"{mode}_barchart_norm_delta.png")
    )


def plot_matplotlib(df, mode, other, path):

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
        [f"argumentation_time_{x}" for x in other] + [f"automl_time_{x}" for x in other]
    ].plot.bar().get_figure().savefig(os.path.join(path, f"time_{mode}.png"))

    labels = df["name"]
    x = np.arange(len(labels))  # the label locations
    width = 0.2  # the width of the bars

    fig, ax = plt.subplots()
    ax.bar(x - (width / 2 * 3), df["baseline_500"], width, label="baseline")
    ax.bar(x - width / 2, df["pkb_500"], width, label="PKB")
    ax.bar(x + width / 2, df["ika_500"], width, label="IKA")
    ax.bar(x + (width / 2 * 3), df["pkb_ika_500"], width, label="PKB + IKA")
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
        os.path.join(path, f"barchart_delta.png"),
        bbox_extra_artists=(lgd, text),
        bbox_inches="tight",
    )

    labels = df["name"]
    x = np.arange(len(labels))  # the label locations
    width = 0.2  # the width of the bars
    fig, ax = plt.subplots()
    ax.bar(x - (width / 2 * 3), df["iteration_baseline_500"], width, label="baseline")
    ax.bar(x - width / 2, df["iteration_pkb_500"], width, label="PKB")
    ax.bar(x + width / 2, df["iteration_ika_500"], width, label="IKA")
    ax.bar(
        x + (width / 2 * 3),
        df["iteration_pkb_ika_500"],
        width,
        label="PKB + IKA",
    )
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
        os.path.join(path, f"barchart_delta_iteration.png"),
        bbox_extra_artists=(lgd, text),
        bbox_inches="tight",
    )


path = os.path.join("/", "home", "results")
df = pd.read_csv(os.path.join(path, "small_baseline_500_summary.csv"))
df = df.set_index("id")
# plot_pd(
#     df,
#     "small_baseline_5000",
#     ["hamlet_250", "hamlet_1000_kb3", "hamlet_250_kb3_fixed"],
#     path,
# )
plot_matplotlib(
    df,
    "baseline_500",
    ["pkb_500", "ika_500", "pkb_ika_500"],
    path,
)
# plot("baseline_1000_218", [], 1000)
# plot("baseline_7200s", ["hamlet_250_new"], 1000)
