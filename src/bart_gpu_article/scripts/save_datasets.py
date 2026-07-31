"""Download, preprocess, and save a few selected OpenML datasets to disk."""

import sys
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace
from collections.abc import Sequence
from pathlib import Path

import jax.numpy as jnp
import polars as pl
from jax import config

from bart_gpu_article.datasim import Data as SimData
from bart_gpu_article.datasim import save_data
from bart_gpu_article.scripts.explore_datasets import (
    process_dataset,
    read_datasets_list,
)

SELECTED_DATASETS = (
    "Australian-Electricity-Demand",
    "Higgs",
    "Radar-Traffic-Data",
    "delays_zurich_transport",
    "poker",
    "sf-police-incidents",
)


def parse_args(argv: Sequence[str]) -> Namespace:
    """Parse command line arguments."""
    parser = ArgumentParser(
        description=__doc__,
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-d",
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="directory to write the datasets into",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrite existing dataset directories",
    )
    return parser.parse_args(argv)


def to_sim_data(X: pl.DataFrame, y: pl.Series) -> SimData:
    """Pack a preprocessed (X, y) pair into a `datasim.Data` for serialization."""
    nan = jnp.float32(jnp.nan)
    return SimData(
        raw_X=jnp.asarray(X.to_numpy().T),
        quantized_X=None,
        y=jnp.asarray(y.to_numpy()),
        max_split=None,
        prior_var=nan,
        pop_var=nan,
        eps_var=nan,
        q=jnp.int32(0),
        binary=jnp.bool_(y.dtype.is_integer()),
    )


def main(argv: Sequence[str] = sys.argv[1:]) -> None:
    """Entry point of the script."""
    config.update("jax_platforms", "cpu")
    args = parse_args(argv)
    datasets = read_datasets_list()
    args.data_dir.mkdir(parents=True, exist_ok=True)
    for name in SELECTED_DATASETS:
        matches = datasets.filter(pl.col("name") == name)
        if matches.height != 1:
            raise RuntimeError(
                f"expected exactly one entry for {name!r}, found {matches.height}"
            )
        meta = matches.row(0, named=True)
        target = args.data_dir / f"dataset-{name}"
        if target.exists() and not args.overwrite:
            print(f"skip {target}, already exists (use --overwrite to redo)...")
            continue
        _, data = process_dataset(meta)
        print(f"save to {target}...")
        simdata = to_sim_data(data.X, data.y)
        save_data(simdata, target, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
