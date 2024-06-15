import shutup; shutup.please()

from automl.utils.argparse import parse_args
from automl.hamlet import run


if __name__ == "__main__":
    args = parse_args()
    run(args)
