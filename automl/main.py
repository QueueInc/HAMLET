import warnings

warnings.filterwarnings("ignore")

from utils.argparse import parse_args
from hamlet.engine import run


if __name__ == "__main__":
    args = parse_args()
    run(args)
