"""Generate and serialize simulated data to disk."""

import sys
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace
from collections.abc import Sequence
from pathlib import Path

from jax import block_until_ready, random

from bart_gpu_article.datasim import make_data, save_data


def parse_args(argv: Sequence[str]) -> Namespace:
    """Parse command line arguments."""
    parser = ArgumentParser(
        description=__doc__,
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-n",
        "--sample-size",
        type=int,
        required=True,
        dest="n",
        help="number of samples",
    )
    parser.add_argument(
        "-p",
        "--num-features",
        type=int,
        required=True,
        dest="p",
        help="number of predictors",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        required=True,
        help="random seed",
    )
    parser.add_argument(
        "-d",
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="directory to write the dataset into",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrite an existing dataset directory",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] = sys.argv[1:]) -> None:
    """Entry point of the script."""
    args = parse_args(argv)
    key = random.key(args.seed)
    data = block_until_ready(
        make_data(key, args.n, args.p, quantized_x="both")
    )
    target = args.data_dir / f"savedata-{args.n}-{args.p}-{args.seed}"
    args.data_dir.mkdir(parents=True, exist_ok=True)
    print(f"save to {target}...")
    save_data(data, target, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
