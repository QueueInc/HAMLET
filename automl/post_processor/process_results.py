import os
import json


def extract_results(path, iteration):
    results = {}
    for root, dirs, files in os.walk(path):
        if f"automl_output_{iteration}.json" in files:
            # Opening JSON file
            with open(
                os.path.join(root, f"automl_output_{iteration}.json")
            ) as json_file:
                results[root.split("/")[3]] = json.load(json_file)

    with open(os.path.join(path, "summary.json"), "w") as outfile:
        json.dump(results, outfile, indent=4)


path = os.path.join("/", "home", "results")
extract_results(os.path.join(path, "baseline"), 1)
extract_results(os.path.join(path, "hamlet"), 4)
