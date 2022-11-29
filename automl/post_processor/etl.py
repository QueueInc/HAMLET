import os
import argparse

import pandas as pd
from plotter import plot_matplotlib
from summarizer import extract_comparison_results, extract_results, summarize_results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Automated Machine Learning Workflow creation and configuration"
    )
    parser.add_argument(
        "-budget",
        "--budget",
        nargs="?",
        type=int,
        required=True,
        help="iteration budget",
    )
    parser.add_argument(
        "-mode",
        "--mode",
        nargs="?",
        type=str,
        required=True,
        help="mode of the experiments",
    )
    args = parser.parse_args()
    return args


def extract(budget, path, hamlet_path):
    extract_results(budget, hamlet_path, "baseline")
    extract_results(budget, hamlet_path, "pkb")
    extract_results(budget, hamlet_path, "ika")
    extract_results(budget, hamlet_path, "pkb_ika")

    extract_comparison_results(
        os.path.join(path, f"auto_sklearn_{budget}"), "auto_sklearn"
    )
    # extract_comparison_results(
    #     os.path.join(path, "auto_sklearn_pro_500"), "auto_sklearn_pro"
    # )
    extract_comparison_results(os.path.join(path, f"h2o_{budget}"), "h2o")


def summarize(budget, path, hamlet_path):
    summarize_results("baseline", ["pkb", "ika", "pkb_ika"], budget, hamlet_path)

    df = pd.read_csv(os.path.join(hamlet_path, "summary.csv")).set_index("id")
    df_auto_sklearn = pd.read_csv(
        os.path.join(path, f"auto_sklearn_{budget}", "summary.csv")
    ).set_index("id")
    # df_auto_sklearn_pro_500 = pd.read_csv(
    #     os.path.join(path, "auto_sklearn_pro_500", "summary.csv")
    # ).set_index("id")
    df_h2o = pd.read_csv(os.path.join(path, f"h2o_{budget}", "summary.csv")).set_index(
        "id"
    )

    others = [
        df_auto_sklearn,
        # df_auto_sklearn_pro_500,
        df_h2o,
    ]

    for x in others:
        df = df.join(x)

    return df


def main(args):
    path = os.path.join("/", "home", "results")
    hamlet_path = os.path.join(path, f"{args.mode}")

    extract(args.budget, path, hamlet_path)
    summary = summarize(args.budget, path, hamlet_path)

    plot_matplotlib(
        summary,
        "baseline",
        [
            "pkb",
            "ika",
            "pkb_ika",
            "auto_sklearn",
            # "auto_sklearn_pro",
            "h2o",
        ],
        hamlet_path,
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
