import os
import json
import pandas as pd


def extract_results(path, iteration):
    results = {}
    for root, dirs, files in os.walk(path):

        if f"automl_output_{iteration}.json" in files:
            results[root.split("/")[4]] = {
                "graph_generation_time": 0,
                "space_generation_time": 0,
                "optimization_time": 0,
                "mining_time": 0,
            }
            for it in range(1, iteration + 1):
                if f"automl_output_{it}.json" in files:
                    # Opening JSON file
                    with open(
                        os.path.join(root, f"automl_output_{it}.json")
                    ) as json_file:
                        loaded = json.load(json_file)
                        if it == iteration:
                            current_json = results[root.split("/")[4]]
                            results[root.split("/")[4]] = loaded
                        else:
                            current_json = loaded
                        results[root.split("/")[4]][
                            "graph_generation_time"
                        ] += current_json["graph_generation_time"]
                        results[root.split("/")[4]][
                            "space_generation_time"
                        ] += current_json["space_generation_time"]
                        results[root.split("/")[4]][
                            "optimization_time"
                        ] += current_json["optimization_time"]
                        results[root.split("/")[4]]["mining_time"] += current_json[
                            "mining_time"
                        ]

    with open(os.path.join(path, "summary.json"), "w") as outfile:
        json.dump(results, outfile, indent=4)


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


def summarize(baseline, other, exhaustive, limit):
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

                if approach != baseline and approach != exhaustive:

                    data[dataset][f"delta_iteration_{approach}"] = (
                        data[dataset][f"iteration_{approach}"]
                        - data[dataset][f"iteration_{baseline}"]
                    )

                    # data[dataset][f"norm_iteration_{approach}"] = round(
                    #     data[dataset][f"delta_iteration_{approach}"]
                    #     / min(1000, data[dataset][f"tot_iteration_{baseline}"]),
                    #     2,
                    # )

                    data[dataset][f"delta_{approach}"] = round(
                        (data[dataset][approach] - data[dataset][baseline]), 3
                    )

                    # if exhaustive in data[dataset]:
                    #     data[dataset][f"norm_{approach}"] = round(
                    #         (
                    #             (
                    #                 data[dataset][f"delta_{approach}"]
                    #                 if abs(data[dataset][f"delta_{approach}"]) > 0.02
                    #                 else 0.0
                    #             )
                    #             / (data[dataset][exhaustive] - data[dataset][baseline])
                    #         ),
                    #         2,
                    #     )
                else:
                    import datetime

                    data[dataset][f"time_{approach}"] = str(
                        datetime.timedelta(seconds=result["optimization_time"])
                    )

    df = pd.DataFrame.from_dict(data, orient="index")
    # .sort_values(
    #     [f"delta_{x}" for x in other], ascending=False
    # )
    mode = f"small_{baseline}"

    mf = pd.read_csv(
        os.path.join("/", "home", "resources", "dataset-meta-features.csv")
    )
    mf["did"] = mf["did"].astype("str")
    mf = mf.set_index("did")
    df = pd.concat([df, mf], axis=1, join="inner")
    df = df.loc[["40983", "40499", "1485", "1478", "1590"]]  # "554"
    df.index.names = ["id"]
    df = df.reset_index()
    df.to_csv(os.path.join(path, f"{mode}_summary.csv"), index=False)

    ## mf = pd.read_csv(os.path.join("resources", "dataset-meta-features.csv"))
    ## mf = mf[(mf["NumberOfInstances"] >= 1000) & (mf["NumberOfFeatures"] >= 50)]
    ## mf["did"] = mf["did"].astype("str")
    ## mf = mf.set_index("did")
    ## df1 = pd.concat([df, mf], axis=1, join="inner")

    ## print_data(df, f"full_{baseline}", other, path)
    ## print_data(df1, f"medium_{baseline}", other, path)
    # print_data(df2, f"small_{baseline}", other, path)


path = os.path.join("/", "home", "results")
extract_results(os.path.join(path, "baseline_500"), 1)
extract_results(os.path.join(path, "pkb_500"), 1)
extract_results(os.path.join(path, "ika_500"), 4)
extract_results(os.path.join(path, "pkb_ika_500"), 4)

summarize(
    "baseline_500",
    ["pkb_500", "ika_500", "pkb_ika_500"],
    "exhaustive",
    500,
)
