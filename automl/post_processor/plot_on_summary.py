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

extract_comparison_results(os.path.join(path, "auto_sklearn_500"), "auto_sklearn")
extract_comparison_results(
    os.path.join(path, "auto_sklearn_pro_500"), "auto_sklearn_pro"
)

summarize("baseline", ["pkb", "ika", "pkb_ika"], 500, hamlet_path)

df = pd.read_csv(os.path.join(hamlet_path, "summary.csv")).set_index("id")
df_auto_sklearn_500 = pd.read_csv(
    os.path.join(path, "auto_sklearn_500", "summary.csv")
).set_index("id")
df_auto_sklearn_pro_500 = pd.read_csv(
    os.path.join(path, "auto_sklearn_pro_500", "summary.csv")
).set_index("id")

others = [df_auto_sklearn_500, df_auto_sklearn_pro_500]

for x in others:
    df = df.join(x)

plot_matplotlib(
    df,
    "baseline",
    ["pkb", "ika", "pkb_ika", "auto_sklearn", "auto_sklearn_pro"],
    hamlet_path,
)
