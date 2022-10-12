from importlib.resources import path
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import false


def get_filtered_datasets(input_df):
    df = pd.read_csv(os.path.join("resources", "dataset-meta-features.csv"))
    df = df.loc[
        df["NumberOfMissingValues"] / (df["NumberOfInstances"] * df["NumberOfFeatures"])
        < 0.1
    ]
    df = df.loc[
        df["NumberOfInstancesWithMissingValues"] / df["NumberOfInstances"] < 0.1
    ]
    df = df.loc[df["NumberOfInstances"] * df["NumberOfFeatures"] < 5000000]
    df = df["did"]
    return input_df[input_df["Unnamed: 0"].isin(df.values.flatten().tolist())]


def plot():

    path = os.path.join("/", "home", "results")
    df = pd.read_csv(os.path.join(path, "summary.csv"))
    df["target_1"] = df.apply(lambda row: row["delta_hamlet_250"] > 0.1, axis=1)
    df["target_3"] = df.apply(lambda row: row["delta_hamlet_250"] > 0.3, axis=1)
    df["target_5"] = df.apply(lambda row: row["delta_hamlet_250"] > 0.5, axis=1)
    df["target_6"] = df.apply(lambda row: row["delta_hamlet_250"] > 0.6, axis=1)
    df = df.set_index("Unnamed: 0")
    mf = pd.read_csv(os.path.join("resources", "dataset-meta-features.csv"))
    mf = mf.set_index("did")
    merged = pd.concat([df, mf], axis=1, join="inner")
    merged.to_csv(os.path.join(path, "merged.csv"))
    merged = merged[list(mf.columns) + ["target_1", "target_3", "target_5", "target_6"]]
    merged.to_csv(os.path.join(path, "merged4weka.csv"), index=False)

    mf = pd.read_csv(os.path.join("resources", "dataset-meta-features.csv"))
    mf = mf[(mf["NumberOfInstances"] >= 1000) & (mf["NumberOfFeatures"] >= 50)]
    mf = mf.set_index("did")
    merged = pd.concat([df, mf], axis=1, join="inner")
    # df = get_filtered_datasets(df)
    # df = df[(df["baseline_5000"] <= 70) & (df["baseline_5000"] >= 30)]

    f = plt.figure()
    df.boxplot([f"delta_{x}" for x in ["hamlet_250"]])
    f.savefig(os.path.join(path, "boxplot_delta.png"))

    f = plt.figure()
    df.boxplot([f"normalized_distance_{x}" for x in ["hamlet_250"]])
    f.savefig(os.path.join(path, "boxplot_nd.png"))

    # df.to_csv(os.path.join(path, "summary.csv"))


# plot("baseline_5000", ["hamlet_250", "hamlet_150"], None)
plot()
