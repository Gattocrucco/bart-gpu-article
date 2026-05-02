"""Plot results saved by `fullbench` script."""

import json
import re
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace
from collections.abc import Sequence
from pathlib import Path
from sys import argv
from typing import NamedTuple

import numpy as np
import polars as pl
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

_METHOD_STYLES = (
    {"markerfacecolor": "black", "markeredgecolor": "none"},
    {"markerfacecolor": "white", "markeredgecolor": "black"},
    {"markerfacecolor": "red", "markeredgecolor": "black"},
)


def load_results(paths: Sequence[Path]) -> pl.DataFrame:
    """Load the given fullbench result files into a single dataframe."""
    tables = []
    for path in paths:
        print(f"read {path}...")
        with open(path) as f:
            tables.append(pl.DataFrame(json.load(f)))
    return pl.concat(tables, how="vertical")


class Agg(NamedTuple):
    df: pl.DataFrame
    device: str
    rounds: int
    n_test: int


def aggregate(df: pl.DataFrame) -> Agg:
    """Aggregate per (method, dataset) and check device/round invariants."""
    agg = df.group_by("method", "dataset_path", maintain_order=True).agg(
        n=pl.col("n_train").first(),
        p=pl.col("p").first(),
        n_rounds=pl.len(),
        mean_rmse=pl.col("rmse").pow(2).mean().sqrt(),
        rmse_sdev=pl.col("rmse").std(),
        mean_logloss=pl.col("logloss").mean(),
        logloss_sdev=pl.col("logloss").std(),
        mean_coverage_50=pl.col("coverage_50").mean(),
        coverage_50_sdev=pl.col("coverage_50").std(),
        mean_time_train=pl.col("time_train").mean(),
        time_train_sdev=pl.col("time_train").std(),
        mean_time_test=pl.col("time_test").mean(),
        time_test_sdev=pl.col("time_test").std(),
    )

    return Agg(
        agg,
        df.get_column("device").unique().item(),
        agg.get_column("n_rounds").unique().item(),
        df.get_column("n_test").unique().item(),
    )


def plot(agg: Agg) -> Figure:
    """Render the aggregated table as dots-with-errorbars per dataset."""
    plt.close("all")
    plt.rcdefaults()

    info_df = agg.df.unique(subset=["dataset_path"], maintain_order=True).select(
        "dataset_path", "n", "p"
    )
    info = {d: (n, p) for d, n, p in info_df.iter_rows()}
    datasets = list(info)
    datasets.sort(key=lambda d: (info[d], d))
    y_pos = {d: i for i, d in enumerate(datasets)}

    methods = sorted(agg.df["method"].unique().to_list())
    offsets = (
        np.linspace(-0.2, 0.2, len(methods)) if len(methods) > 1 else np.array([0.0])
    )

    panels = (
        (f"RMSE (n_test={agg.n_test})", "rmse"),
        ("log-loss\n(bayes. and class. only)", "logloss"),
        ("50% coverage\n(bayes. regr. only)", "coverage_50"),
        ("train time [s]", "time_train"),
        ("predict time [s]", "time_test"),
    )

    fig, axes = plt.subplots(
        1,
        len(panels),
        sharey=True,
        figsize=[10, 0.7 * len(datasets) + 2.0],
        num="fullbench-plot",
        clear=True,
        layout="constrained",
    )

    for ax, (xlabel, col) in zip(axes, panels):
        for method, dy, style in zip(methods, offsets, _METHOD_STYLES):
            sub = agg.df.filter(pl.col("method") == method)
            ys = [y_pos[d] + dy for d in sub["dataset_path"]]
            ax.errorbar(
                sub[f"mean_{col}"].to_numpy(),
                ys,
                xerr=sub[f"{col}_sdev"].to_numpy(),
                fmt="o",
                markersize=10,
                capsize=4,
                color="black",
                label=method,
                **style,
            )
        ax.set_xlabel(xlabel)
        ax.grid(linestyle="--", axis="x")
        if col == "coverage_50":
            ax.axvline(0.5, color="black", linestyle="--")

    def _label(path: str) -> str:
        name = Path(path).name
        if re.fullmatch(r"savedata(-[^-]+){4}", name):
            name = "<simulated>"
        return f"{name}\nn={info[path][0]:_}\np={info[path][1]:_}"

    axes[0].set_yticks(range(len(datasets)))
    axes[0].set_yticklabels([_label(d) for d in datasets])
    axes[0].set_ylim(-1, len(datasets) - 0.4)
    axes[0].invert_yaxis()

    for ax in axes:
        ax.grid(axis="y", linestyle=":")

    for ax in axes[3:]:
        ax.set(xscale="log")
        ax.minorticks_on()
        ax.yaxis.set_minor_locator(plt.NullLocator())
        ax.grid(which="minor", linestyle=":")

    axes[-1].legend(loc="upper right")
    fig.suptitle(f"device: {agg.device}")
    fig.supxlabel(f"+/– sdev over {agg.rounds} rounds", fontsize="medium")

    return fig


def save_figure(fig: Figure) -> None:
    """Save the figure under ./plots as PDF."""
    outdir = Path("./plots")
    outdir.mkdir(exist_ok=True)
    file = outdir / f"{fig.get_label()}.pdf"
    print(f"write {file}...")
    fig.savefig(file)


def parse_args(argv_: Sequence[str]) -> Namespace:
    """Parse command line arguments."""
    parser = ArgumentParser(
        description=__doc__,
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "files",
        nargs="+",
        type=Path,
        help="result files produced by fullbench",
    )
    return parser.parse_args(argv_)


def main(argv: Sequence[str] = argv[1:]) -> None:
    """Entry point of the script."""
    args = parse_args(argv)
    df = load_results(args.files)
    agg = aggregate(df)
    fig = plot(agg)
    save_figure(fig)
    plt.show()


if __name__ == "__main__":
    main()
