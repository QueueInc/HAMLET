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

    labels = list(df.index)
    x = np.arange(len(labels))  # the label locations
    width = 0.25  # the width of the bars

    fig, ax = plt.subplots()
    ax.bar(x - width, df["delta_baseline_1000_kb3"], width, label="s.k.b.")
    ax.bar(x, df["delta_hamlet_250"], width, label="i.k.a.")
    ax.bar(x + width, df["delta_hamlet_250_kb3"], width, label="s.k.b. + i.k.a")
    ax.set_ylabel("Improvement")
    ax.set_title("Improvement of balanced accuracy w.r.t. the baseline")
    ax.set_xticks(x, labels)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(path, f"barchart_delta.png"))

    labels = list(df.index)
    x = np.arange(len(labels))  # the label locations
    width = 0.25  # the width of the bars
    fig, ax = plt.subplots()
    ax.bar(
        x - width, abs(df["delta_iteration_baseline_1000_kb3"]), width, label="s.k.b."
    )
    ax.bar(x, abs(df["delta_iteration_hamlet_250"]), width, label="i.k.a.")
    ax.bar(
        x + width,
        abs(df["delta_iteration_hamlet_250_kb3"]),
        width,
        label="s.k.b. + i.k.a",
    )
    ax.set_ylabel("Iterations")
    ax.set_title(
        "Number of configurations ahead to find\nthe best result w.r.t. the baseline"
    )
    ax.set_xticks(x, labels)
    ax.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(path, f"barchart_delta_iteration.png"))


path = os.path.join("/", "home", "results")
df = pd.read_csv(os.path.join(path, "small_baseline_5000_summary.csv"))
df = df.set_index("id")
plot_pd(
    df,
    "small_baseline_5000",
    ["hamlet_250", "baseline_1000_kb3", "hamlet_250_kb3"],
    path,
)
plot_matplotlib(
    df,
    "small_baseline_5000",
    ["hamlet_250", "baseline_1000_kb3", "hamlet_250_kb3"],
    path,
)
# plot("baseline_1000_218", [], 1000)
# plot("baseline_7200s", ["hamlet_250_new"], 1000)
