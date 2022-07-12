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

    df[
        [f"argumentation_time_{x}" for x in other] + [f"automl_time_{x}" for x in other]
    ].plot.bar().get_figure().savefig(os.path.join(path, f"time_{mode}.png"))

    labels = df["name"]
    x = np.arange(len(labels))  # the label locations
    width = 0.2  # the width of the bars

    fig, ax = plt.subplots()
    ax.bar(x - (width / 2 * 3), df["baseline_5000"], width, label="baseline")
    ax.bar(x - width / 2, df["hamlet_1000_kb3"], width, label="PKB")
    ax.bar(x + width / 2, df["hamlet_250"], width, label="IKA")
    ax.bar(x + (width / 2 * 3), df["hamlet_250_kb3_fixed"], width, label="PKB + IKA")
    ax.set_ylabel("Balanced accuracy")
    ax.set_title("Balanced accuracy achieved by the approaches")
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
        bbox_to_anchor=(0.5, -0.05),
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
    ax.bar(x - (width / 2 * 3), df["iteration_baseline_5000"], width, label="baseline")
    ax.bar(x - width / 2, df["iteration_hamlet_1000_kb3"], width, label="PKB")
    ax.bar(x + width / 2, df["iteration_hamlet_250"], width, label="IKA")
    ax.bar(
        x + (width / 2 * 3),
        df["iteration_hamlet_250_kb3_fixed"],
        width,
        label="PKB + IKA",
    )
    ax.set_ylabel("No. configuration")
    ax.set_title("No. configuration in which the approaches achieve the best result")
    ax.set_xticks(x, labels)
    # ax.legend()

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    lgd = fig.legend(
        by_label.values(),
        by_label.keys(),
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, -0.05),
    )
    text = fig.text(-0.2, 1.05, "", transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig(
        os.path.join(path, f"barchart_delta_iteration.png"),
        bbox_extra_artists=(lgd, text),
        bbox_inches="tight",
    )


path = os.path.join("/", "home", "results")
df = pd.read_csv(os.path.join(path, "small_baseline_5000_summary.csv"))
df = df.set_index("id")
# plot_pd(
#     df,
#     "small_baseline_5000",
#     ["hamlet_250", "hamlet_1000_kb3", "hamlet_250_kb3_fixed"],
#     path,
# )
plot_matplotlib(
    df,
    "small_baseline_5000",
    ["hamlet_250", "hamlet_1000_kb3", "hamlet_250_kb3_fixed"],
    path,
)
# plot("baseline_1000_218", [], 1000)
# plot("baseline_7200s", ["hamlet_250_new"], 1000)
