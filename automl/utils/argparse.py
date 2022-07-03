import argparse


def parse_args():

    parser = argparse.ArgumentParser(description="HAMLET")

    parser.add_argument(
        "-dataset",
        "--dataset",
        nargs="?",
        type=str,
        required=True,
        help="dataset to analise",
    )
    parser.add_argument(
        "-metric",
        "--metric",
        nargs="?",
        type=str,
        required=True,
        help="metric to optimize",
    )
    parser.add_argument(
        "-mode",
        "--mode",
        nargs="?",
        type=str,
        required=True,
        help="either minimize or maximize the metric (min or max)",
    )
    parser.add_argument(
        "-batch_size",
        "--batch_size",
        nargs="?",
        type=int,
        required=True,
        help="num iterations to visit",
    )
    parser.add_argument(
        "-time_budget",
        "--time_budget",
        nargs="?",
        type=int,
        required=True,
        help="time budget in seconds",
    )
    parser.add_argument(
        "-input_path",
        "--input_path",
        nargs="?",
        type=str,
        required=True,
        help="path to the automl input",
    )
    parser.add_argument(
        "-output_path",
        "--output_path",
        nargs="?",
        type=str,
        required=True,
        help="path to the automl ouput",
    )
    parser.add_argument(
        "-seed",
        "--seed",
        nargs="?",
        type=int,
        required=True,
        help="seed for reproducibility",
    )

    args = parser.parse_args()
    return args
