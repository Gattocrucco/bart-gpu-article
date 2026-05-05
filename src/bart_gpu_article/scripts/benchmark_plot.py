"""Plot the results of `benchmark`."""

import json
import sys
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace
from pathlib import Path
from typing import Sequence

import polars as pl
from cycler import Cycler
from labellines import labelLines
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from bart_gpu_article.textbox import textbox


def load_results() -> list[dict]:
    """Load all benchmark result JSON files from ./results directory."""
    results_dir = Path("./results")
    results = []
    for filepath in sorted(results_dir.glob("benchmark*.json")):
        print(f"read {filepath}...")
        with open(filepath) as f:
            results.append(json.load(f))
    return results


def results_to_df(results: list[dict], filter: bool) -> pl.DataFrame:
    """Merge results into a dataframe."""
    tables = []
    for things in results:
        tables.append(
            pl.DataFrame(things["results"]).with_columns(
                [pl.lit(v).alias(k) for k, v in things.items() if k != "results"]
            )
        )

    df = (
        pl.concat(tables, how="diagonal")
        .with_columns(
            pl.col("device_kind").replace(
                {
                    "NVIDIA L4": "L4",
                    "NVIDIA A100-SXM4-40GB": "A100",
                    "NVIDIA RTX A4000": "A4000",
                    "NVIDIA RTX PRO 6000 Blackwell Workstation Edition": "P6000",
                    "NVIDIA RTX PRO 5000 Blackwell": "P5000",
                    "NVIDIA GeForce RTX 5060 Ti": "5060Ti",
                    "Apple M1 Pro": "M1pro",
                    "AMD EPYC 7402 24-Core Processor": "epyc6",
                }
            )
        )
        .with_columns(case=pl.concat_str("package", "device_kind", separator="-"))
    )

    if filter:
        df = df.filter(
            # keep only one cpu and one gpu, exclude the rest
            pl.col("device_kind").is_in(["A4000", "P6000", "5060Ti", "epyc6"]).not_(),
            # skip catboost cpu, keep only gpu, just to reduce clutter
            (pl.col("package") == "catboost")
            .and_(pl.col("device_kind") == "M1pro")
            .not_(),
        )

    return df


def get_cycler() -> Cycler:
    """Return a cycler of properties for plotting lines."""
    return plt.cycler(
        color=[
            "#006BA4",
            "#FF800E",
            "#ABABAB",
            "#595959",
            "#5F9ED1",
            "#C85200",
            "#898989",
            "#A2C8EC",
            "#FFBC79",
            "#CFCFCF",
        ][:5],  # from style tableau-colorblind10
        # color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'][:5],
        # color=5 * ['#c00'],
        linestyle=["-", "--", "-.", ":", "-"],
        marker=5 * ["."],
        markerfacecolor=5 * ["none"],
    )


def plot(df: pl.DataFrame, single_figure: bool):
    """Generate speed benchmark plots."""
    # reset matplotlib
    plt.close("all")
    plt.rcdefaults()

    keynames = ["n/ntree", "ntree", "n/p", "p"]
    groups = list(df.group_by(keynames, maintain_order=True))

    if single_figure:
        fig, axs = plt.subplots(
            2,
            2,
            figsize=[8.5, 8.5],
            num="benchmark-plot",
            clear=True,
            layout="constrained",
            sharex=True,
            sharey=True,
        )
        axs = axs.flat
        figs = [fig]
    else:
        axs = []
        figs = []
        for i in range(len(groups)):
            fig, ax = plt.subplots(
                figsize=[4.5, 4],
                num=f"benchmark-plot-{i}",
                clear=True,
                layout="constrained",
            )
            axs.append(ax)
            figs.append(fig)
    axs[1], axs[3] = axs[3], axs[1]

    cycler = get_cycler()

    if single_figure:
        ax = axs[0]
        # set log scale before plotting to allow labelLines correct auto positioning
        ax.set(xscale="log", yscale="log")

    for ax, (keys, group) in zip(axs, groups):
        ax.set_prop_cycle(cycler)
        for (case,), data in group.sort("case").group_by("case", maintain_order=True):
            ax.plot(data["n"], data["time_per_iter"], label=case)

        ss = ax.get_subplotspec()

        if ss.is_last_row():
            ax.set_xlabel("n")
        if ss.is_first_col():
            ax.set_ylabel("Time per iteration [s]")

        if not single_figure:
            ax.set(xscale="log", yscale="log")
            ax.set(xlim=(10, None))
            xvals = None

        handtuning_label_ordering = [
            "bartz-P5000",
            "bartz-M1pro",
            "dbarts-M1pro",
            "xgboost-P5000",
            "xgboost-M1pro",
            "catboost-P5000",
        ]

        match keys:
            case (_, None, _, None):  # top left
                xvals = [100_000, 3500, 300, 3000, 100, 100]
            case (None, _, None, _):  # top right
                xvals = [100, 100, 200, 1000, 1000, 100]
            case (None, _, _, None):  # bottom left
                xvals = [100, 100, 200, 1000, 6000, 100]
            case (_, None, None, _):  # bottom right
                xvals = [100, 3500, 300, 400, 6000, 100]

        labels = [line.get_label() for line in ax.get_lines()]
        if set(labels) != set(handtuning_label_ordering):
            print("final selection not recognized, skip hand-tuning labels")
            xvals = None
        else:
            # re-sort to match actual lines
            xvals = dict(zip(handtuning_label_ordering, xvals))
            xvals = [xvals[k] for k in labels]

        labelLines(ax.get_lines(), xvals=xvals, outline_width=3)

        ax.grid(linestyle="--")
        ax.grid(which="minor", linestyle=":")

        textbox(
            ax,
            "\n".join(
                f"{name}={value}"
                for name, value in zip(keynames, keys)
                if value is not None
            ),
            loc="upper left",
        )

    if single_figure:
        ax = axs[0]
        # set xlim after plotting to use "None" for auto limit
        ax.set_xlim(10, None)

    save_figures(figs)

    # show figures interactively
    plt.show()


def save_figures(figs: list[Figure]) -> None:
    """Save all figures in ./plots as PDF."""
    outdir = Path("./plots")
    outdir.mkdir(exist_ok=True)
    for fig in figs:
        file = outdir / f"{fig.get_label()}.pdf"
        print(f"write {file}...")
        fig.savefig(file)


def parse_args(argv: Sequence[str]) -> Namespace:
    """Parse command line arguments."""
    parser = ArgumentParser(
        description=__doc__,
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-s",
        "--single-figure",
        action="store_true",
        dest="single_figure",
        help="combine all plots into a single figure",
    )
    parser.add_argument(
        "-f",
        "--filter",
        action="store_true",
        help="keep only the selected data for the final plot",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] = sys.argv[1:]):
    args = parse_args(argv)
    results = load_results()
    df = results_to_df(results, args.filter)
    plot(df, args.single_figure)


if __name__ == "__main__":
    main()
