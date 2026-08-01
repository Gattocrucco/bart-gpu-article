"""Plot results saved by `fullbench` script."""

import json
import re
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace
from collections.abc import Sequence
from itertools import chain
from math import lcm
from pathlib import Path
from sys import argv
from typing import NamedTuple

import numpy as np
import polars as pl
from labellines import labelLine
from labellines.utils import normalize_xydata
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.layout_engine import TightLayoutEngine

DEVICE_NICKNAMES = {
    "NVIDIA_RTX_PRO_5000_Blackwell": "RTX PRO 5000",
    "NVIDIA_GeForce_RTX_3060": "RTX 3060",
    "Apple_M1_Pro": "M1 Pro",
}

METHOD_NICKNAMES = {"bartzadaptive": "bartz+"}

OUTCOME_NICKNAMES = {"continuous": "regr.", "binary": "class."}

METHOD_STYLES = (
    {"markerfacecolor": "black", "markeredgecolor": "none"},
    {"markerfacecolor": "white", "markeredgecolor": "black"},
    {"markerfacecolor": "red", "markeredgecolor": "black"},
)

MARKERSIZE = 6  # points
DOT_SHIFT = 0.138  # vertical gap between dots of the same dataset, in data units,
# tuned by eye such that vertically aligned dots touch
FIGSIZE = [8.5, 11.5]  # inches; tight layout is not adaptive, so this is set to
# fill a page of the article: the width matches the other full-page figures, and
# the height leaves room for a caption of about 8 lines


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
        # failed runs are recorded as all-None rows, so skip the nulls
        outcome_type=pl.col("outcome_type").drop_nulls().first(),
        n_rounds=pl.len(),
        mean_relmse=(pl.col("rmse") / pl.col("test_sdev")).pow(2).mean(),
        relmse_sdev=(pl.col("rmse") / pl.col("test_sdev")).pow(2).std(),
        mean_logloss=pl.col("logloss").mean(),
        logloss_sdev=pl.col("logloss").std(),
        mean_coverage_50=pl.col("coverage_50").mean(),
        coverage_50_sdev=pl.col("coverage_50").std(),
        mean_time_train=pl.col("time_train").mean(),
        time_train_sdev=pl.col("time_train").std(),
        mean_time_test=pl.col("time_test").mean(),
        time_test_sdev=pl.col("time_test").std(),
        mean_num_trees=pl.col("num_trees").mean(),
        num_trees_sdev=pl.col("num_trees").std(),
        # the inverse of the acceptance is the mean number of proposals per
        # accepted move, which spreads out the low-acceptance datasets
        mean_inv_move_acc=(1 / pl.col("move_acc")).mean(),
        inv_move_acc_sdev=(1 / pl.col("move_acc")).std(),
        mean_mean_leaves=pl.col("mean_leaves").mean(),
        mean_leaves_sdev=pl.col("mean_leaves").std(),
    )

    return Agg(
        agg,
        df.get_column("device").unique().item(),
        agg.get_column("n_rounds").unique().item(),
        df.get_column("n_test").unique().item(),
    )


def _minor_grid(ax: plt.Axes) -> None:
    """Draw the minor grid on the x axis only, leaving the y axis alone."""
    ax.yaxis.set_minor_locator(plt.NullLocator())
    ax.grid(which="minor", linestyle=":")


def plot(agg: Agg) -> Figure:
    """Render the aggregated table as dots-with-errorbars per dataset."""
    plt.close("all")
    plt.rcdefaults()

    info_df = agg.df.unique(subset=["dataset_path"], maintain_order=True).select(
        "dataset_path", "n", "p"
    )
    info = {d: (n, p) for d, n, p in info_df.iter_rows()}
    outcome = dict(
        agg.df.group_by("dataset_path")
        .agg(pl.col("outcome_type").drop_nulls().first())
        .iter_rows()
    )
    datasets = list(info)
    datasets.sort(key=lambda d: (info[d], d))
    y_pos = {d: i for i, d in enumerate(datasets)}

    methods = sorted(agg.df["method"].unique().to_list())
    offsets = (np.arange(len(methods)) - (len(methods) - 1) / 2) * DOT_SHIFT

    # (xlabel, aggregated column, width ratio), one tuple per panel row
    panel_rows = (
        (
            (f"MSE / test var (n_test={agg.n_test:_})", "relmse", 2),
            ("log-loss", "logloss", 1),
            ("coverage", "coverage_50", 1),
        ),
        (
            ("train time [s]", "time_train", 2),
            ("predict time [s]", "time_test", 1),
        ),
        (
            ("number of trees", "num_trees", 1),
            ("1 / move acceptance", "inv_move_acc", 1),
            ("leaves per tree", "mean_leaves", 1),
        ),
    )

    # lay the panels out on a grid with enough columns that each panel spans a
    # whole number of them, proportional to its width ratio
    ncols = lcm(*(sum(w for *_, w in row) for row in panel_rows))
    mosaic = [
        [col for _, col, w in row for _ in range(w * ncols // total)]
        for row, total in ((row, sum(w for *_, w in row)) for row in panel_rows)
    ]

    fig, axd = plt.subplot_mosaic(
        mosaic,
        sharey=True,
        figsize=FIGSIZE,
        num="fullbench-plot",
        clear=True,
        # constrained layout gives up on a grid with this many columns, and
        # tight layout does too unless the default padding is reduced
        layout=TightLayoutEngine(pad=0.5, w_pad=-1),
    )

    ref_lines = []  # labelled after the axis limits are final

    for xlabel, col, _ in chain.from_iterable(panel_rows):
        ax = axd[col]
        for method, dy, style in zip(methods, offsets, METHOD_STYLES):
            sub = agg.df.filter(pl.col("method") == method)
            ys = [y_pos[d] + dy for d in sub["dataset_path"]]
            ax.errorbar(
                sub[f"mean_{col}"].to_numpy(),
                ys,
                xerr=sub[f"{col}_sdev"].to_numpy(),
                fmt="o",
                markersize=MARKERSIZE,
                capsize=4,
                color="black",
                label=METHOD_NICKNAMES.get(method, method),
                **style,
            )
        ax.set_xlabel(xlabel)
        ax.grid(linestyle="--", axis="x")
        ax.grid(axis="y", linestyle=":")

        if col == "relmse":
            # logit scale: differences represent MSE ratios on high SNR
            # datasets, and explained variance ratios on noisy datasets
            ax.set_xscale("logit")
            lo = (agg.df["mean_relmse"] - agg.df["relmse_sdev"]).min()
            hi = (agg.df["mean_relmse"] + agg.df["relmse_sdev"]).max()
            ax.set_xlim(
                10 ** np.floor(np.log10(lo)),
                1 - 10 ** np.floor(np.log10(1 - hi)),
            )
            ax.xaxis.set_minor_formatter(plt.NullFormatter())
            _minor_grid(ax)
            # the legend title doubles as the info box of the whole figure
            device = DEVICE_NICKNAMES.get(agg.device, agg.device.replace("_", " "))
            ax.legend(
                loc="upper right",
                title=f"{device}\n$\\pm$ sdev {agg.rounds} rounds",
                title_fontsize="medium",  # matches the legend entries
            )
        elif col == "coverage_50":
            ax.set_xlim(0, 1)
            ref_lines.append(
                ax.axvline(0.5, color="black", linestyle="--", label="target 50%")
            )
        elif col in ("time_train", "time_test"):
            ax.set_xscale("log")
            ax.minorticks_on()
            _minor_grid(ax)
        elif col == "num_trees":
            ax.set_xlim(left=0)
        elif col == "inv_move_acc":
            # these two quantities are bounded below by 1, but the axes start at
            # 0 to make distances along them read as ratios
            ax.set_xlim(left=0)
        elif col == "mean_leaves":
            ax.set_xlim(left=0)
            ref_lines += [
                ax.axvline(32, color="black", linestyle="--", label="bartz max"),
                ax.axvline(64, color="black", linestyle="--", label="xgboost max"),
            ]

    def _label(path: str) -> str:
        name = Path(path).name
        if re.fullmatch(r"savedata(-[^-]+){4}", name):
            name = "<simulated>"
        else:
            # dataset names come in inconsistent flavors of Capitalized-Words
            # and lower_case_words
            name = name.removeprefix("dataset-")
            name = name.replace("-", " ").replace("_", " ").lower()
        kind = OUTCOME_NICKNAMES[outcome[path]]
        return f"{name}\nn={info[path][0]:_}, p={info[path][1]:_}, {kind}"

    # the y axis is shared, so this applies to all panels
    ax = axd[panel_rows[0][0][1]]
    ax.set_yticks(range(len(datasets)))
    ax.set_yticklabels([_label(d) for d in datasets])
    ax.set_ylim(-0.6, len(datasets) - 0.4)
    ax.invert_yaxis()

    # labelLine freezes the label position, so it must run last. It matches the
    # requested x against the line coordinates put through a transform round
    # trip, which for a vertical line is an exact comparison that the round trip
    # breaks, so feed it the round-tripped value.
    for line in ref_lines:
        x, y = normalize_xydata(line)
        # y[0] and y[1] are the bottom and top ends of the line; labelLine would
        # put the label at their midpoint, so offset it up to the top edge. The
        # y axis is inverted, so the aligned text reads downwards: anchor its
        # start and it hangs from the top edge.
        y_top = y[1] + 0.02 * (y[0] - y[1])
        labelLine(line, x[0], yoffset=y_top - y.mean(), ha="left", outline_width=4)

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
