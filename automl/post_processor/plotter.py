from importlib.resources import path
import json
import os
import pandas as pd
import matplotlib.pyplot as plt


def get_best_in(target, evaluated_rewards):
    filtered = [
        x["balanced_accuracy"] if x["balanced_accuracy"] != "-inf" else 0
        for x in evaluated_rewards[:target]
    ]
    return max(filtered)


def get_position(target, evaluated_rewards):
    if target == 0:
        return -1
    filtered = [
        x["balanced_accuracy"] if x["balanced_accuracy"] != "-inf" else 0
        for x in evaluated_rewards
        if x["status"] != "previous_constraint"
    ]
    return next((i for i, x in enumerate(filtered) if x >= target), -1)


def print_data(df, mode, other, path):
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

    df.to_csv(os.path.join(path, f"{mode}_summary.csv"))


def plot(baseline, other, exhaustive, limit):
    data = {}
    path = os.path.join("/", "home", "results")
    for approach in [baseline, exhaustive] + other:
        with open(os.path.join(path, approach, "summary.json")) as f:
            for dataset, result in json.load(f).items():
                if approach == baseline:
                    data[dataset] = {}
                elif dataset not in data:
                    continue

                temp = (
                    result["best_config"]["balanced_accuracy"]
                    if limit is None or approach == exhaustive
                    else get_best_in(limit, result["evaluated_rewards"])
                )
                data[dataset][approach] = round(temp * 100, 2)
                data[dataset][f"iteration_{approach}"] = get_position(
                    temp, result["evaluated_rewards"]
                )

                data[dataset][f"tot_iteration_{approach}"] = len(
                    [
                        x
                        for x in result["evaluated_rewards"]
                        if x["status"] != "previous_constraint"
                    ]
                )

                if approach != baseline and approach != exhaustive:

                    data[dataset][f"delta_iteration_{approach}"] = (
                        data[dataset][f"iteration_{approach}"]
                        - data[dataset][f"iteration_{baseline}"]
                    )

                    data[dataset][f"norm_iteration_{approach}"] = round(
                        data[dataset][f"delta_iteration_{approach}"]
                        / min(1000, data[dataset][f"tot_iteration_{baseline}"]),
                        2,
                    )

                    data[dataset][f"delta_{approach}"] = round(
                        (data[dataset][approach] - data[dataset][baseline]), 2
                    )

                    if exhaustive in data[dataset]:
                        data[dataset][f"norm_{approach}"] = round(
                            (
                                (
                                    data[dataset][f"delta_{approach}"]
                                    if abs(data[dataset][f"delta_{approach}"]) > 2
                                    else 0.0
                                )
                                / (data[dataset][exhaustive] - data[dataset][baseline])
                            ),
                            2,
                        )
                else:
                    import datetime

                    data[dataset][f"time_{approach}"] = str(
                        datetime.timedelta(seconds=result["optimization_time"])
                    )

    df = pd.DataFrame.from_dict(data, orient="index").sort_values(
        [f"delta_{x}" for x in other], ascending=False
    )

    # mf = pd.read_csv(os.path.join("resources", "dataset-meta-features.csv"))
    # mf = mf[(mf["NumberOfInstances"] >= 1000) & (mf["NumberOfFeatures"] >= 50)]
    # mf["did"] = mf["did"].astype("str")
    # mf = mf.set_index("did")
    # df1 = pd.concat([df, mf], axis=1, join="inner")

    df2 = df.loc[["40983", "40499", "1485", "1478", "1590"]]  # "554"
    # print_data(df, f"full_{baseline}", other, path)
    # print_data(df1, f"medium_{baseline}", other, path)
    print_data(df2, f"small_{baseline}", other, path)


plot(
    "baseline_5000",
    ["hamlet_250", "baseline_1000_kb3", "hamlet_250_kb3"],
    "exhaustive",
    1000,
)
# plot("baseline_1000_218", [], 1000)
# plot("baseline_7200s", ["hamlet_250_new"], 1000)
