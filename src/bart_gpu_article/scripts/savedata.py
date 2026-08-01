"""Generate and serialize simulated data to disk."""

import sys
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace
from collections.abc import Sequence
from pathlib import Path

from jax import block_until_ready, config, random

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
        "-q",
        "--num-quadratic",
        type=int,
        default=None,
        dest="q",
        help="number of quadratic interaction terms (default: 2 if p > 2 else 0)",
    )
    parser.add_argument(
        "--binary",
        action="store_true",
        help="generate binary (probit) outcomes instead of continuous",
    )
    parser.add_argument(
        "--peff",
        type=int,
        default=None,
        help="effective number of active predictors (SpikeSlab importance"
        " scales); default: all predictors equally important",
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
    config.update("jax_platforms", "cpu")
    args = parse_args(argv)
    key = random.key(args.seed)
    data = block_until_ready(
        make_data(
            key,
            args.n,
            args.p,
            quantized_x="both",
            q=args.q,
            binary=args.binary,
            peff=args.peff,
        )
    )
    outcome = "binary" if args.binary else "continuous"
    peff = "" if args.peff is None else f"-peff{args.peff}"
    target = args.data_dir / f"savedata-{args.n}-{args.p}-{int(data.q)}-{outcome}{peff}"
    args.data_dir.mkdir(parents=True, exist_ok=True)
    print(f"save to {target}...")
    save_data(data, target, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
