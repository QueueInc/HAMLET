import os

import pandas as pd
from plotter import plot_matplotlib
from summarizer import extract_results, summarize


path = os.path.join("/", "home", "results", "ext_500")
extract_results(os.path.join(path, "baseline"), 1)
extract_results(os.path.join(path, "pkb"), 1)
extract_results(os.path.join(path, "ika"), 4)
extract_results(os.path.join(path, "pkb_ika"), 4)

summarize("baseline", ["pkb", "ika", "pkb_ika"], 500, path)

df = pd.read_csv(os.path.join(path, "summary.csv"))
df = df.set_index("id")
plot_matplotlib(
    df,
    "baseline",
    ["pkb", "ika", "pkb_ika"],
    path,
)
