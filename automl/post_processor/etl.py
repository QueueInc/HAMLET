import os
import argparse

import pandas as pd
from plotter import plot_matplotlib, time_plot
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
        "-input-folder",
        "--input-folder",
        nargs="?",
        type=str,
        required=True,
        help="mode of the experiments",
    )
    parser.add_argument(
        "-output-folder",
        "--output-folder",
        nargs="?",
        type=str,
        required=True,
        help="mode of the experiments",
    )
    args = parser.parse_args()
    return args


def extract(budget, path, input_folder, output_folder):
    extract_results(budget, path, input_folder, output_folder, "baseline")
    extract_results(budget, path, input_folder, output_folder, "pkb")
    extract_results(budget, path, input_folder, output_folder, "ika")
    extract_results(budget, path, input_folder, output_folder, "pkb_ika")

    extract_comparison_results(
        os.path.join(path, f"auto_sklearn_{budget}"), "auto_sklearn"
    )
    # extract_comparison_results(
    #     os.path.join(path, "auto_sklearn_pro_500"), "auto_sklearn_pro"
    # )
    extract_comparison_results(os.path.join(path, f"h2o_pro_{budget}"), "h2o")


def summarize(budget, path, output_folder):
    summarize_results(
        "baseline", ["pkb", "ika", "pkb_ika"], budget, os.path.join(path, output_folder)
    )

    df = pd.read_csv(os.path.join(path, output_folder, "summary.csv")).set_index("id")
    df_auto_sklearn = pd.read_csv(
        os.path.join(path, f"auto_sklearn_{budget}", "summary.csv")
    ).set_index("id")
    # df_auto_sklearn_pro_500 = pd.read_csv(
    #     os.path.join(path, "auto_sklearn_pro_500", "summary.csv")
    # ).set_index("id")
    df_h2o = pd.read_csv(
        os.path.join(path, f"h2o_pro_{budget}", "summary.csv")
    ).set_index("id")

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

    if not os.path.exists(os.path.join(path, args.output_folder)):
        os.makedirs(os.path.join(path, args.output_folder))

    extract(args.budget, path, args.input_folder, args.output_folder)
    summary = summarize(args.budget, path, args.output_folder)

    plot_matplotlib(
        summary,
        "baseline",
        [
            "pkb",
            "ika",
            "pkb_ika",
        ],
        [
            "auto_sklearn",
            "h2o",
        ],
        os.path.join(path, args.output_folder),
    )
    time_plot(summary, os.path.join(path, args.output_folder), args.budget, "time")
    time_plot(summary, os.path.join(path, args.output_folder), args.budget, "iteration")


if __name__ == "__main__":
    args = parse_args()
    main(args)
