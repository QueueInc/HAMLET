from functools import reduce
import os
import datetime
import json
from numpy import int32
from operator import itemgetter
import pandas as pd
from sklearn.metrics import balanced_accuracy_score


def merge_results(
    current_iteration, results, current_json, threshold, mode, dataset, tot_iterations
):
    def common_elements(list1, list2):
        return [element for element in list1 if element in list2]

    if current_iteration == 1:
        # Add iteration and conf number in evaluated_rewards (useful to plot marker in accuracy_time)
        for idx, reward in enumerate(current_json["evaluated_rewards"]):
            reward["iteration"] = current_iteration
            reward["conf_numb"] = idx
        # If it is the first iteration I instantiate the results (w/ the laoded json, filtering afterwards)
        results = current_json
    else:
        # Otherwise, I start by summing the timing (useful to plot marker in accuracy_time)
        results["graph_generation_time"] += current_json["graph_generation_time"]
        results["space_generation_time"] += current_json["space_generation_time"]
        results["optimization_time"] += current_json["optimization_time"]
        results["mining_time"] += current_json["mining_time"]
        # I take the distinct set of all the generated rules (I think I cannot do better than this)
        results["rules"] += [
            rule for rule in current_json["rules"] if rule not in results["rules"]
        ]
        # I find the common elements between the visisted confs
        common_elems = common_elements(
            results["points_to_evaluate"], current_json["points_to_evaluate"]
        )
        if common_elems:
            # If there are some, the index from which I should start copying is the index of the last common element
            start = current_json["points_to_evaluate"].index(common_elems[-1]) + 1
        else:
            # Otherwise, I copy everything
            start = 0
        # I copy everything in that range
        # Add iteration and conf number in evaluated_rewards (useful to plot marker in accuracy_time)
        for idx, reward in enumerate(current_json["evaluated_rewards"][start:]):
            reward["iteration"] = current_iteration
            reward["conf_numb"] = idx
        results["points_to_evaluate"] += current_json["points_to_evaluate"][start:]
        results["evaluated_rewards"] += current_json["evaluated_rewards"][start:]

    # I also take track of the time iteration by iteration
    results[f"start_time_{current_iteration}"] = current_json["start_time"]
    results[f"graph_generation_time_{current_iteration}"] = current_json[
        "graph_generation_time"
    ]
    results[f"space_generation_time_{current_iteration}"] = current_json[
        "space_generation_time"
    ]
    results[f"optimization_time_{current_iteration}"] = current_json[
        "optimization_time"
    ]
    results[f"mining_time_{current_iteration}"] = current_json["mining_time"]

    # I take the index of the elements to keep (based on the time filtering)
    if current_iteration < tot_iterations:
        return results

    if mode != "pkb_ika" and mode != "ika":
        cut_off_index = next(
            (
                index
                for index, elem in enumerate(results["evaluated_rewards"])
                if (elem["absolute_time"] if "absolute_time" in elem else float("-inf"))
                - results["start_time"]
                >= threshold
            ),
            len(results["evaluated_rewards"]) - 1,
        )

        results["evaluated_rewards"] = results["evaluated_rewards"][:cut_off_index]
        results["points_to_evaluate"] = results["points_to_evaluate"][:cut_off_index]

    # if mode == "ika" and dataset == "40499":
    #     print(results["evaluated_rewards"])

    # I calculate the index of the best config among the evaluaed rewards
    best_index = max(
        range(len(results["evaluated_rewards"])),
        key=lambda index: float(
            results["evaluated_rewards"][index]["balanced_accuracy"]
        ),
    )

    # if mode == "ika" and dataset == "40499":
    #     print(results["evaluated_rewards"][best_index])

    # I put tat config as the best one
    results["best_config"] = results["evaluated_rewards"][best_index].copy()
    results["best_config"]["config"] = results["points_to_evaluate"][best_index]
    results["best_config"]["time"] = (
        reduce(
            lambda x, y: x + (y["total_time"] if "total_time" in y else 0),
            results["evaluated_rewards"][:best_index],
            0,
        )
        / 60
    )

    # print(f"{mode} {dataset}: {results['best_config']}")
    return results
    # temp_list = [
    #     (index, elem)
    #     for index, elem in enumerate(results["evaluated_rewards"])
    #     if "absolute_time" in elem
    #     and elem["absolute_time"] - results["start_time"] < threshold
    # ]

    # I filter the evaluated rewards
    # results["evaluated_rewards"] = [elem for _, elem in temp_list]
    # if temp_list:
    #     # If temp list is not empty, I filter the points to evaluate
    #     results["points_to_evaluate"] = list(
    #         itemgetter(*[index for index, _ in temp_list])(
    #             results["points_to_evaluate"]
    #         )
    #     )
    #     # I transform the evaluated rewards that are -inf to float
    #     for d in results["evaluated_rewards"]:
    #         d.update(
    #             (k, float(v))
    #             for k, v in d.items()
    #             if k == "balanced_accuracy" and isinstance(v, str) and v == "-inf"
    #         )
    #     # I calculate the index of the best config among the evaluaed rewards
    #     best_index = max(
    #         range(len(results["evaluated_rewards"])),
    #         key=lambda index: results["evaluated_rewards"][index]["balanced_accuracy"],
    #     )
    #     print(f"dataset: {dataset}, mode: {mode}, index: {best_index}")
    #     # I put tat config as the best one
    #     results["best_config"] = results["evaluated_rewards"][best_index].copy()
    #     results["best_config"]["config"] = results["points_to_evaluate"][best_index]
    # else:
    #     results["points_to_evaluate"].clear()
    # return results


def extract_results(budget, path, input_folder, output_folder, mode):
    results = {}
    iteration = 1 if mode in ("baseline", "pkb") else (4 if budget == 500 else 8)
    threshold = (
        float("inf")
        if mode in ("ika", "pkb_ika")
        else (3600 if budget == 500 else 7200)
    )

    for root, _, files in os.walk(os.path.join(path, input_folder, mode)):

        if f"automl_output_{iteration}.json" in files:
            dataset_id = root.split("/")[-3]
            results[dataset_id] = {}
            for it in range(1, iteration + 1):
                if f"automl_output_{it}.json" in files:
                    # Opening JSON file
                    with open(
                        os.path.join(root, f"automl_output_{it}.json")
                    ) as json_file:
                        loaded = json.load(json_file)
                    results[dataset_id] = merge_results(
                        current_iteration=it,
                        results=results[dataset_id],
                        current_json=loaded,
                        threshold=threshold,
                        mode=mode,
                        dataset=dataset_id,
                        tot_iterations=iteration,
                    )

    if not os.path.exists(os.path.join(path, output_folder, mode)):
        os.makedirs(os.path.join(path, output_folder, mode))
    with open(os.path.join(path, output_folder, mode, "summary.json"), "w") as outfile:
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
                f"best_time_{label}": 0,
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


def summarize_results(baseline, other, limit, output_path):
    data = {}
    for approach in [baseline] + other:
        with open(os.path.join(output_path, approach, "summary.json")) as f:
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

                data[dataset][f"best_time_{approach}"] = result["best_config"]["time"]

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
    df.to_csv(os.path.join(output_path, "summary.csv"), index=False)
