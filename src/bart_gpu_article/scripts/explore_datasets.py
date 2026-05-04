"""Explore the OpenML datasets selected by `list_datasets`."""

import argparse
import sys
from collections.abc import Sequence
from numbers import Number
from pathlib import Path
from types import MappingProxyType
from typing import Any, NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from openml import OpenMLDataset
from openml.datasets import get_dataset
from rapidfuzz import fuzz, process

DATASET_TARGETS = MappingProxyType(
    {
        "2018-Airplane-Flights": "PricePerTicket",
        "5-years-historical-stock-quotes": "close_price",
        "Australian-Electricity-Demand": "value_0",
        "BOT-IoT": "attack",
        "Covid19-us": "value_0",  # confirmed cases
        "FitBit_HeartRate": "Value",
        "FitBit_Steps": "Steps",
        "M4-competition-daily": "value_0",
        "Methane": "MM256",
        "New-York-Citi-Bike-Trip-Duration-2016": "trip_duration",
        "Radar-Traffic-Data": "Volume",
        "Solar-Power": "value_0",
        "Wind-Power": "value_0",
        "cleaned-Edge-IIoTset": "Attack_label",
        "freMTPL2freq": "ClaimNb",
    }
)

def print_data_summary(x: pl.DataFrame, y: pl.Series) -> None:
    with pl.Config(
        fmt_str_lengths=100, tbl_cols=-1, tbl_width_chars=10_000, tbl_rows=-1
    ):
        print("\nX & y (last column)")
        x = x.with_columns(y)
        x.glimpse()
        descr = x.describe()
        xnu = x.select(pl.lit("n_unique").alias("statistic"), pl.all().n_unique())
        descr = pl.concat([descr, xnu], how="vertical_relaxed")
        descr = descr.transpose(
            include_header=True, header_name="Column", column_names="statistic"
        ).drop("count", "null_count")
        print(descr)


def print_categorical_predictors_info(X: pl.DataFrame):
    print("\n--- Categorical variables (unique value counts) ---")
    cat_cols = [col for col, dtype in X.schema.items() if dtype == pl.Categorical]

    if not cat_cols:
        print("  (none)")
    else:
        for col in cat_cols:
            n_unique = X[col].n_unique()
            values_preview = sorted(X[col].drop_nulls().unique().to_list())[:10]
            preview_str = str(values_preview) + (" ..." if n_unique > 10 else "")
            print(f"  {col}: {n_unique} unique values  {preview_str}")


def read_datasets_list() -> pl.DataFrame:
    file = Path("./data/list-datasets.csv")
    print(f"read {file}...")
    return pl.read_csv(file)


def parse_argv_and_filter_datasets(
    argv: Sequence[str], datasets: pl.DataFrame
) -> tuple[pl.DataFrame, bool]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-n",
        "--name",
        metavar="NAME",
        default=None,
        help="Select a single dataset by name (fuzzy match).",
    )
    parser.add_argument(
        "-f",
        "--from",
        dest="from_",
        action="store_true",
        default=False,
        help="Process all datasets starting from the one matched by -n (requires -n).",
    )
    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        default=False,
        help=(
            "Populate module globals `original_data` and `data` with the processed "
            "dataset, for use in interactive ipython sessions (requires -n)."
        ),
    )
    args = parser.parse_args(argv)

    if args.from_ and args.name is None:
        parser.error("-f/--from requires -n/--name to be specified.")

    if args.interactive and args.name is None:
        parser.error("-i/--interactive requires -n/--name to be specified.")

    if args.name is not None:
        names = datasets["name"].to_list()
        result = process.extractOne(args.name, names, scorer=fuzz.WRatio)
        if result is None:
            parser.error(f"No dataset found matching {args.name!r}.")
        matched_name, score, _ = result
        print(f"Matched dataset: {matched_name!r} (score={score:.0f})")
        if args.from_:
            idx = names.index(matched_name)
            datasets = datasets.slice(idx)
        else:
            datasets = datasets.filter(pl.col("name") == matched_name)

    return datasets, args.interactive


def to_sin_cos(expr: pl.Expr, period: Number | pl.Expr) -> tuple[pl.Expr, pl.Expr]:
    """Map a periodic value to its sine and cosine."""
    t = expr * (2 * np.pi / period)
    return t.sin().name.suffix("_sin"), t.cos().name.suffix("_cos")


def main(argv: Sequence[str] = sys.argv[1:]) -> None:
    datasets = read_datasets_list()
    datasets, interactive = parse_argv_and_filter_datasets(argv, datasets)
    for meta in datasets.iter_rows(named=True):
        original_data, data = process_dataset(meta, log=True)
        if interactive:
            globals()["original_data"] = original_data
            globals()["data"] = data


def basic_checks(X: pl.DataFrame, y: pl.Series) -> None:
    """check there are no null values and other things."""
    assert X.null_count().sum_horizontal().item() == 0
    assert y is not None
    assert y.null_count() == 0
    assert y.name not in X.columns


class Data(NamedTuple):
    """Output of `get_data`."""

    dataset: OpenMLDataset
    X: pl.DataFrame
    y: pl.Series


def get_data(name: str, did: int) -> Data:
    """Get dataset from OpenML and do basic preprocessing."""
    print("download data...")
    dataset = get_dataset(
        did,
        download_data=True,
        error_if_multiple=True,
        cache_format="feather",
        download_qualities=True,
        download_features_meta_data=True,
    )
    assert dataset.name == name

    print("extract data...")
    target = dataset.default_target_attribute
    if target is not None and "," in target:
        # multiple targets, not supported by get_data
        target = None
    X_pd, y_pd, cat, cols = dataset.get_data(target)

    # convert to polars
    X = pl.DataFrame(X_pd)
    if y_pd is None:
        y = None
    else:
        y = pl.Series(y_pd)

    # check columns
    assert np.all(X.columns == cols)
    del cols

    # check categorical columns
    for col, dtype, is_cat in zip(X.columns, X.dtypes, cat):
        assert is_cat == (dtype == pl.Categorical), col
    del cat

    # pick target for datasets that don't come with a single default target
    if dataset.name in DATASET_TARGETS:
        assert y is None
        y = X[DATASET_TARGETS[dataset.name]]
        targets = {y.name}
        if dataset.default_target_attribute is not None:
            other_targets = dataset.default_target_attribute.split(",")
            assert len(other_targets) > 1
            targets.update(other_targets)
    else:
        assert y is not None
        targets = set()

    # remove all targets from X
    X = X.drop(targets)

    return Data(dataset, X, y)


def custom_preprocessing(data: Data) -> Data:
    """Bespoke preprocessing for each dataset."""
    dataset, X, y = data

    match dataset.name:
        case "2018-Airplane-Flights":
            X = X.with_columns(
                *to_sin_cos(pl.col("Quarter"), 4),
            ).drop(
                "Unnamed:_0",  # this is just an index
                "ItinID",  # this is a coarser version of MktID, redundant
                "Quarter",  # already converted to periodic form
            )

        case "Covid19-us":
            X = X.drop(
                "value_1",  # deaths
            )

        case "delays_zurich_transport":
            X = (
                X.with_columns(
                    pl.col("direction").cast(pl.String).str.to_integer(),
                    *to_sin_cos(pl.col("weekday").cast(pl.String).str.to_integer(), 7),
                    *to_sin_cos(pl.col("dayminute"), 24 * 60),
                )
                .drop(
                    "hour",  # redundant with dayminute
                    "weekday",  # already converted to periodic form
                    "dayminute",  # already converted to periodic form
                )
                .to_dummies(["vehicle_type"])
            )

        case "Higgs" | "poker":
            # everything already ok
            pass

        case _:
            print(f"==== No custom pre-processing defined for {dataset.name} ====")

    return Data(dataset, X, y)


def process_dataset(
    meta: dict[str, Any], *, log: bool = False
) -> tuple[Data, Data]:
    """Process a single dataset, `meta` is one row in the list of datasets.

    With ``log=True``, also print summaries of X and y and write the
    y-distribution plot.
    """
    did = meta["did"]
    print(f"\n\n####### DATASET {meta['name']} (id {did}) #######")

    original_data = get_data(**meta)
    dataset, X, original_y = custom_preprocessing(original_data)
    ystuff = preprocess_y(original_y)
    y = ystuff.y
    basic_checks(X, y)
    if log:
        print_y_info(original_y, ystuff)
        print_data_summary(X, y)
        print_categorical_predictors_info(X)
        plot_y_distribution(did, dataset, ystuff)
    return original_data, Data(dataset, X, y)


class YStuff(NamedTuple):
    y: pl.Series
    n_unique: int
    is_binary: bool


def preprocess_y(y: pl.Series) -> YStuff:
    """Put the target column in a standard format and do some checks."""
    y_n_unique = y.n_unique()

    # determine type of y
    is_binary = y.dtype in [pl.Categorical, pl.String] or (
        y.dtype.is_integer() and y_n_unique == 2
    )

    # check type determination is consistent and put y in a standard format
    if is_binary:
        assert y_n_unique == 2
        y_unique = y.unique().sort().to_list()
        y = y.replace_strict(y_unique, [0, 1], return_dtype=pl.Int8)
    else:  # continuous
        assert y_n_unique > 4
        assert y.dtype.is_numeric()
        y = y.cast(pl.Float64)

    return YStuff(y, y_n_unique, is_binary)


def print_y_info(original_y: pl.Series, ystuff: YStuff) -> None:
    """Print analysis of the target variable."""
    print("\n--- Target variable (y) analysis ---")

    cutoff = 10
    y_unique_cut = original_y.unique().sort()[:cutoff].to_list()
    if ystuff.n_unique > cutoff:
        y_unique_cut.append(f"... (other {ystuff.n_unique - cutoff})")
    print(f"y.dtype={original_y.dtype}")
    print(f"unique values: {y_unique_cut}")

    if ystuff.is_binary:
        print(ystuff.y.value_counts(normalize=True))


def plot_y_distribution(
    did: int,
    dataset: OpenMLDataset,
    ystuff: YStuff,
) -> None:
    """Plot the histogram of y, save the figure to file."""
    # reset matplotlib
    plt.close("all")
    plt.rcdefaults()

    fig, ax = plt.subplots(
        figsize=(8, 4), clear=True, layout="constrained", num=f"explore_datasets-{did}"
    )
    y_np = ystuff.y.to_numpy()

    # --- choose bins and data to plot ---
    if ystuff.n_unique <= 100:
        # few unique values: discrete bins centered on each value
        bins = ystuff.y.unique().cast(pl.Int64).sort().to_numpy()
        mid = (bins[:-1] + bins[1:]) / 2
        left = bins[0] - (mid[0] - bins[0])
        right = bins[-1] + (bins[-1] - mid[-1])
        bins = np.concatenate([[left], mid, [right]]).tolist()
        y_plot = y_np
    else:
        # many unique values: clip to quantiles 0.1–99.9
        q_lo = np.percentile(y_np, 0.1)
        q_hi = np.percentile(y_np, 99.9)
        y_plot = y_np[(y_np >= q_lo) & (y_np <= q_hi)]
        bins = "auto"

    counts, bin_edges, _ = ax.hist(y_plot, bins=bins, histtype="step")

    # --- decoration depending on variable type ---
    if ystuff.is_binary:
        # annotate each bar with its fraction and decimal value
        n_total = len(y_np)
        for i, count in enumerate(counts):
            count = int(count)
            if count > 0:
                frac = count / n_total
                bin_center = (bin_edges[i] + bin_edges[i + 1]) / 2
                # place label above the line if <= 50%, below if > 50%
                if frac <= 0.5:
                    xytext = (0, 4)
                    va = "bottom"
                else:
                    xytext = (0, -4)
                    va = "top"
                ax.annotate(
                    f"{count}/{n_total} ({frac:#.2g})",
                    xy=(bin_center, count),
                    xytext=xytext,
                    textcoords="offset points",
                    ha="center",
                    va=va,
                    fontsize="small",
                )
    else:  # continuous
        # draw quantile vertical lines, annotated near the top
        q_levels = np.array([0.1, 1, 10, 50, 90, 99, 99.9])
        q_values = np.percentile(y_np, q_levels)
        cmap = plt.colormaps["viridis"]
        colors = cmap(np.linspace(0.2, 0.8, len(q_levels)))
        for idx_q, (ql, qv, c) in enumerate(zip(q_levels, q_values, colors)):
            ax.axvline(
                qv,
                color=c,
                linestyle="--",
                linewidth=0.8,
                alpha=0.8,
            )
            ax.annotate(
                f"{ql:g}%",
                xy=(qv, 1),
                xycoords=("data", "axes fraction"),
                xytext=(2, -4 - idx_q * 10),
                textcoords="offset points",
                fontsize="x-small",
                color=c,
                va="top",
                ha="left",
                rotation=0,
            )

        # annotate min and max as separate texts at the bottom
        y_min, y_max = float(np.min(y_np)), float(np.max(y_np))
        ax.text(
            0.01,
            0.02,
            f"min = {y_min:#.2g}",
            transform=ax.transAxes,
            fontsize="small",
            verticalalignment="bottom",
            horizontalalignment="left",
        )
        ax.text(
            0.99,
            0.02,
            f"max = {y_max:#.2g}",
            transform=ax.transAxes,
            fontsize="small",
            verticalalignment="bottom",
            horizontalalignment="right",
        )

    # log scale on y-axis
    ax.set_yscale("log")

    # grid with minor ticks on y-axis
    ax.yaxis.set_minor_locator(plt.LogLocator(subs="auto"))
    ax.grid(which="major", axis="y", linewidth=0.8, alpha=0.5)
    ax.grid(which="minor", axis="y", linewidth=0.4, alpha=0.3)

    # decorate plot
    ax.set(
        title=f"Distribution of y — {dataset.name} (id={did})",
        xlabel="y",
        ylabel="Count",
    )

    # save figure to file
    hist_dir = Path("./plots/explore_datasets")
    hist_dir.mkdir(parents=True, exist_ok=True)
    hist_path = hist_dir / f"{did}_{dataset.name}.pdf"
    print(f"write {hist_path}...")
    fig.savefig(hist_path)


if __name__ == "__main__":
    main()
