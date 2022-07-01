from importlib.resources import path
import json
import os
import pandas as pd

data = {}
path = os.path.join("/", "home", "results")
for approach in ["baseline", "hamlet"]:
    with open(os.path.join(path, approach, "summary.json")) as f:
        for dataset, result in json.load(f).items():
            if dataset not in data:
                data[dataset] = {}
            data[dataset][approach] = result["best_config"]["balanced_accuracy"]


pd.DataFrame.from_dict(data, orient="index").to_csv(os.path.join(path, "summary.csv"))
