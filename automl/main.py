import shutup; shutup.please()

from hamlet.utils.argparse import parse_args
from hamlet import run


if __name__ == "__main__":
    args = parse_args()
    run(args)
