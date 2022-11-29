import os
import datetime
import json
from numpy import int32
import pandas as pd
from sklearn.metrics import balanced_accuracy_score


def merge_results(results, current_json):
    def common_elements(list1, list2):
        return [element for element in list1 if element in list2]

    results["start_time"] = min(results["start_time"], current_json["start_time"])
    results["graph_generation_time"] += current_json["graph_generation_time"]
    results["space_generation_time"] += current_json["space_generation_time"]
    results["optimization_time"] += current_json["optimization_time"]
    results["mining_time"] += current_json["mining_time"]
    results["best_config"] = current_json["best_config"]
    results["rules"] += current_json["rules"]
    common_elements = common_elements(
        results["points_to_evaluate"], current_json["points_to_evaluate"]
    )
    start_index = current_json["points_to_evaluate"].index(common_elements[-1]) + 1
    for i in range(start_index, len(current_json["points_to_evaluate"])):
        results["points_to_evaluate"].append(current_json["points_to_evaluate"][i])
        results["evaluated_rewards"].append(current_json["evaluated_rewards"][i])
    return results


def extract_results(path, iteration):
    results = {}
    for root, _, files in os.walk(path):

        if f"automl_output_{iteration}.json" in files:
            dataset_id = root.split("/")[-3]
            for it in range(1, iteration + 1):
                if f"automl_output_{it}.json" in files:
                    # Opening JSON file
                    with open(
                        os.path.join(root, f"automl_output_{it}.json")
                    ) as json_file:
                        loaded = json.load(json_file)
                    if it == 1:
                        results[dataset_id] = loaded
                    else:
                        results[dataset_id] = merge_results(
                            results=results[dataset_id],
                            current_json=loaded,
                        )

    with open(os.path.join(path, "summary.json"), "w") as outfile:
        json.dump(results, outfile, indent=4)


def extract_comparison_results(path, label):
    results = {}
    csv_file_name = (
        "cv_results.csv" if label.startswith("auto_sklearn") else "raw_cv_results.csv"
    )
    for root, dirs, files in os.walk(path):

        if csv_file_name in files:
            dataset_id = root.split("/")[-1]
            cv_results = pd.read_csv(os.path.join(root, csv_file_name))
            if label.startswith("auto_sklearn"):
                cv_results = cv_results.sort_values("rank_test_scores")
                accuracy = cv_results.iloc[0]["mean_test_score"]
                iteration = int(cv_results.iloc[0]["Unnamed: 0"])
            else:
                accuracy = (
                    cv_results.groupby(by="fold")
                    .apply(lambda x: balanced_accuracy_score(x["class"], x["predict"]))
                    .mean()
                )
                iteration = 0
            results[dataset_id] = {
                label: accuracy,
                f"iteration_{label}": iteration,
            }

    with open(os.path.join(path, "summary.json"), "w") as outfile:
        json.dump(results, outfile, indent=4)
    pd.DataFrame(results).T.reset_index().rename(columns={"index": "id"}).to_csv(
        os.path.join(path, "summary.csv"), index=False
    )


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


def summarize_results(baseline, other, limit, path):
    data = {}
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
                data[dataset][approach] = round(temp, 3)
                data[dataset][f"argumentation_time_{approach}"] = (
                    result["graph_generation_time"] + result["space_generation_time"]
                )
                data[dataset][f"automl_time_{approach}"] = (
                    result["optimization_time"] + result["mining_time"]
                )
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

                if approach != baseline:

                    data[dataset][f"delta_iteration_{approach}"] = (
                        data[dataset][f"iteration_{approach}"]
                        - data[dataset][f"iteration_{baseline}"]
                    )

                    data[dataset][f"delta_{approach}"] = round(
                        (data[dataset][approach] - data[dataset][baseline]), 3
                    )

                else:
                    data[dataset][f"time_{approach}"] = str(
                        datetime.timedelta(seconds=result["optimization_time"])
                    )

    df = pd.DataFrame.from_dict(data, orient="index")

    mf = pd.read_csv(
        os.path.join("/", "home", "resources", "dataset-meta-features.csv")
    )
    mf["did"] = mf["did"].astype("str")
    mf = mf.set_index("did")
    df = pd.concat([df, mf], axis=1, join="inner")
    df = df.loc[["40983", "40499", "1485", "1478", "1590"]]  # "554"
    df.index.names = ["id"]
    df = df.reset_index()
    df.to_csv(os.path.join(path, "summary.csv"), index=False)
