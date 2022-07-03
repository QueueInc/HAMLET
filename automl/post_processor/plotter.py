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


data = {}
path = os.path.join("/", "home", "results")
for approach in ["baseline_5000", "hamlet_250", "hamlet_150"]:
    with open(os.path.join(path, approach, "summary.json")) as f:
        for dataset, result in json.load(f).items():
            if dataset not in data:
                data[dataset] = {}
            # data[dataset][approach] = result["best_config"]["balanced_accuracy"]
            data[dataset][approach] = round(
                (get_best_in(1000, result["evaluated_rewards"]) * 100), 2
            )

            if approach != "baseline_5000":
                data[dataset][f"delta_{approach}"] = round(
                    (data[dataset][approach] - data[dataset]["baseline_5000"]), 2
                )
                data[dataset][f"normalized_distance_{approach}"] = round(
                    (
                        data[dataset][f"delta_{approach}"]
                        / (1 - data[dataset]["baseline_5000"])
                    ),
                    2,
                )
            else:
                import datetime

                data[dataset]["iterations"] = len(result["evaluated_rewards"])
                data[dataset]["time"] = str(
                    datetime.timedelta(seconds=result["optimization_time"])
                )

df = pd.DataFrame.from_dict(data, orient="index").sort_values(
    "delta_hamlet_250", ascending=False
)
# df = df[(df["baseline_5000"] <= 70) & (df["baseline_5000"] >= 30)]

f = plt.figure()
df.boxplot(["delta_hamlet_250", "delta_hamlet_150"])
f.savefig(os.path.join(path, "boxplot_delta.png"))

f = plt.figure()
df.boxplot(["normalized_distance_hamlet_250", "normalized_distance_hamlet_150"])
f.savefig(os.path.join(path, "boxplot_nd.png"))

df.to_csv(os.path.join(path, "summary.csv"))
