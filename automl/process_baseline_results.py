import os
import json


results = {}
for root, dirs, files in os.walk("/home/baseline_results"):
    if "automl_output_1.json" in files:
        # Opening JSON file
        with open(os.path.join(root, "automl_output_1.json")) as json_file:
            results[root.split("/")[3]] = json.load(json_file)

with open("/home/baseline_results/summary.json", "w") as outfile:
    json.dump(results, outfile, indent=4)

    # for dir in dirs
    #  for file in files:
    #     with open(os.path.join(root, file), "r") as auto:
