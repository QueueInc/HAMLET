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


def plot(baseline, other, limit):
    data = {}
    path = os.path.join("/", "home", "results")
    for approach in [baseline] + other:
        with open(os.path.join(path, approach, "summary.json")) as f:
            for dataset, result in json.load(f).items():
                if approach == baseline:
                    data[dataset] = {}
                elif dataset not in data:
                    continue

                temp = (
                    result["best_config"]["balanced_accuracy"]
                    if limit is None
                    else get_best_in(limit, result["evaluated_rewards"])
                )
                data[dataset][approach] = round(temp * 100, 2)
                data[dataset][f"iteration_{approach}"] = get_position(
                    temp, result["evaluated_rewards"]
                )

                if approach != baseline:

                    data[dataset][f"iteration_even_{approach}"] = get_position(
                        (data[dataset][baseline] - 2) / 100, result["evaluated_rewards"]
                    )

                    # if data[dataset][f"iteration_even_{approach}"] != -1:
                    data[dataset][f"delta_iteration_{approach}"] = (
                        data[dataset][f"iteration_{approach}"]
                        - data[dataset][f"iteration_{baseline}"]
                    )

                    data[dataset][f"delta_{approach}"] = round(
                        (data[dataset][approach] - data[dataset][baseline]), 2
                    )
                    # data[dataset][f"normalized_distance_{approach}"] = round(
                    #     (
                    #         data[dataset][f"delta_{approach}"]
                    #         / (1 - data[dataset][baseline])
                    #     ),
                    #     2,
                    # )
                else:
                    import datetime

                    data[dataset]["iterations"] = len(result["evaluated_rewards"])
                    data[dataset]["time"] = str(
                        datetime.timedelta(seconds=result["optimization_time"])
                    )

    df = pd.DataFrame.from_dict(data, orient="index").sort_values(
        [f"delta_{x}" for x in other], ascending=False
    )
    mf = pd.read_csv(os.path.join("resources", "dataset-meta-features.csv"))
    mf = mf[(mf["NumberOfInstances"] >= 1000) & (mf["NumberOfFeatures"] >= 50)]
    mf = mf.set_index("did")
    df = pd.concat([df, mf], axis=1, join="inner")
    # df = df[(df["baseline_5000"] <= 70) & (df["baseline_5000"] >= 30)]

    f = plt.figure()
    df.boxplot([f"delta_{x}" for x in other])
    f.savefig(os.path.join(path, "boxplot_delta.png"))

    f = plt.figure()
    df.boxplot([f"delta_iteration_{x}" for x in other])
    f.savefig(os.path.join(path, "boxplot_delta_iteration.png"))

    # f = plt.figure()
    # df.boxplot([f"normalized_distance_{x}" for x in other])
    # f.savefig(os.path.join(path, "boxplot_nd.png"))

    df.to_csv(os.path.join(path, "summary.csv"))


plot("baseline_5000", ["hamlet_250_kb2", "hamlet_250"], 1000)
# plot("baseline_7200s", ["hamlet_250_new"], 1000)
