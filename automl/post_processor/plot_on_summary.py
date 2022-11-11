import os

import pandas as pd
from plotter import plot_matplotlib
from summarizer import extract_comparison_results, extract_results, summarize


path = os.path.join("/", "home", "results")
hamlet_path = os.path.join(path, "ext_500")
extract_results(os.path.join(hamlet_path, "baseline"), 1)
extract_results(os.path.join(hamlet_path, "pkb"), 1)
extract_results(os.path.join(hamlet_path, "ika"), 4)
extract_results(os.path.join(hamlet_path, "pkb_ika"), 4)

extract_comparison_results(os.path.join(path, "auto_sklearn_500"))
extract_comparison_results(os.path.join(path, "auto_sklearn_pro_500"))

summarize("baseline", ["pkb", "ika", "pkb_ika"], 500, path)

df = pd.read_csv(os.path.join(hamlet_path, "summary.csv"))
df_auto_sklearn_500 = pd.read_csv(os.path.join(path, "auto_sklearn_500", "summary.csv"))
df_auto_sklearn_pro_500 = pd.read_csv(
    os.path.join(path, "auto_sklearn_pro_500", "summary.csv")
)
df = df.set_index("id")
plot_matplotlib(
    df,
    [df_auto_sklearn_500],
    "baseline",
    ["pkb", "ika", "pkb_ika"],
    hamlet_path,
)
