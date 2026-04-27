"""Plot results saved by `fullbench` script."""

import json
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace
from collections.abc import Sequence
from pathlib import Path
from sys import argv

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


def aggregate(df: pl.DataFrame) -> tuple[pl.DataFrame, str, int]:
    """Aggregate per (method, dataset) and check device/round invariants."""
    devices = df["device"].unique().to_list()
    assert len(devices) == 1, f"expected single device, got {devices}"

    agg = df.group_by("method", "dataset_path", maintain_order=True).agg(
        n=pl.col("n_train").first(),
        p=pl.col("p").first(),
        n_rounds=pl.len(),
        mean_rmse=pl.col("rmse").pow(2).mean().sqrt(),
        rmse_sdev=pl.col("rmse").std(),
        mean_time_train=pl.col("time_train").mean(),
        time_train_sdev=pl.col("time_train").std(),
        mean_time_test=pl.col("time_test").mean(),
        time_test_sdev=pl.col("time_test").std(),
    )

    rounds = agg["n_rounds"].unique().to_list()
    assert len(rounds) == 1, f"expected same number of rounds per group, got {rounds}"

    return agg, devices[0], rounds[0]


def plot(agg: pl.DataFrame, device: str, n_rounds: int) -> Figure:
    """Render the aggregated table as dots-with-errorbars per dataset."""
    plt.close("all")
    plt.rcdefaults()

    info_df = agg.unique(subset=["dataset_path"], maintain_order=True).select(
        "dataset_path", "n", "p"
    )
    info = {d: (n, p) for d, n, p in info_df.iter_rows()}
    datasets = list(info)
    datasets.sort(key=lambda d: (info[d], d))
    y_pos = {d: i for i, d in enumerate(datasets)}

    methods = sorted(agg["method"].unique().to_list())
    offsets = (
        np.linspace(-0.2, 0.2, len(methods)) if len(methods) > 1 else np.array([0.0])
    )

    fig, ax = plt.subplots(
        figsize=[6.5, 0.7 * len(datasets) + 2.0],
        num="fullbench-plot",
        clear=True,
        layout="constrained",
    )

    for method, dy, style in zip(methods, offsets, _METHOD_STYLES):
        sub = agg.filter(pl.col("method") == method)
        ys = [y_pos[d] + dy for d in sub["dataset_path"]]
        ax.errorbar(
            sub["mean_rmse"].to_numpy(),
            ys,
            xerr=sub["rmse_sdev"].to_numpy(),
            fmt="o",
            markersize=10,
            capsize=4,
            color="black",
            label=method,
            **style,
        )

    ax.set_yticks(range(len(datasets)))
    ax.set_yticklabels(
        [f"{Path(d).name}\n(n={info[d][0]}, p={info[d][1]})" for d in datasets]
    )
    ax.set_ylim(-0.6, len(datasets) - 0.4)
    ax.invert_yaxis()
    ax.set_xlabel(f"RMSE +/- sdev over {n_rounds} rounds")
    ax.set_title(f"device: {device}")
    ax.legend(title="method", loc="best")
    ax.grid(linestyle="--", axis="x")

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
    agg, device, n_rounds = aggregate(df)
    fig = plot(agg, device, n_rounds)
    save_figure(fig)
    plt.show()


if __name__ == "__main__":
    main()
