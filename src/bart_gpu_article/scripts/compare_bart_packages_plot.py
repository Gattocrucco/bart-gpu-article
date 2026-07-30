"""Plot results saved by `compare-bart-packages` script."""

import json
import math
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace
from collections.abc import Sequence
from pathlib import Path
from sys import argv

import polars as pl
from cycler import Cycler
from labellines import labelLines
from matplotlib import pyplot as plt


def load_results() -> list[dict]:
    """Load results saved by `compare-bart-packages`."""
    results_dir = Path("./results")
    all_results = []
    for file in sorted(results_dir.glob("compare-bart-packages*.json")):
        with open(file) as f:
            data = json.load(f)
            all_results.extend(data)
    return all_results


def get_cycler() -> Cycler:
    """Get a cycler of plot properties."""
    return plt.cycler(
        color=[
            "#006BA4",
            "#FF800E",
            "#ABABAB",
            "#595959",
        ],  # from style tableau-colorblind10
        linestyle=["-", "--", "-.", ":"],
        marker=["o", "s", "^", "D"],
    )


def results_to_df(results: list[dict]) -> pl.DataFrame:
    """Merge results into a dataframe."""
    tables = []
    for things in results:
        tables.append(
            pl.DataFrame(things["results"])
            .drop("p", "ntree")
            .with_columns(
                [pl.lit(v).alias(k) for k, v in things.items() if k != "results"]
            )
        )

    df = pl.concat(tables, how="diagonal_relaxed")
    # df = df.filter(pl.col("n") >= 32)
    return df


def plot(df: pl.DataFrame, single_figure: bool) -> list[plt.Figure]:
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
            num="compare-bart-packages-plot",
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
                figsize=[4.5, 3.5],
                num=f"compare-bart-packages-plot-{i}",
                clear=True,
                layout="constrained",
            )
            axs.append(ax)
            figs.append(fig)
    axs[1], axs[3] = axs[3], axs[1]

    cycler = get_cycler()

    for ax, (keys, group) in zip(axs, groups):
        # plot rmse curves
        ax.set_prop_cycle(cycler)
        for (package,), data in group.group_by(["package"], maintain_order=True):
            ax.plot(data["n"], data["rmse"], markerfacecolor="none", label=package)

        # prepare data to plot standard deviation references
        sdev_labels = {
            "eps_var": "error sdev",
            "pop_var": "population sdev",
            "prior_var": "prior sdev",
        }
        vd_check = group.group_by("n").agg(pl.col(*sdev_labels).n_unique())
        assert vd_check.drop("n").select(pl.all_horizontal(pl.all() == 1).all()).item()
        vd = (
            group.group_by("n")
            .agg(
                pl.col(*sdev_labels).first().sqrt(),
            )
            .sort("n")
        )

        # plot standard deviation references
        error_lines = []
        for key, label in sdev_labels.items():
            (line,) = ax.plot(vd["n"], vd[key], "--k", label=label)
            error_lines.append(line)

        # set plot properties that better be set before plotting labels
        ax.set_xscale("log")
        ref_n = df["n"] if single_figure else vd["n"]
        ax.set_xlim(
            10 ** math.floor(math.log10(ref_n.min())),
            10 ** math.ceil(math.log10(ref_n.max())),
        )

        # add labels on top of standard deviation lines
        labelLines(
            error_lines,
            xvals=[20, 2000, 300],
            drop_label=True,
            outline_width=6,
            align=False,
        )

        # add legend; possibly abuse the legend as generic box with text
        ss = ax.get_subplotspec()
        legend_title = "\n".join(
            f"{name}={value}"
            for name, value in zip(keynames, keys)
            if value is not None
        )
        legend_kw = dict(
            title=legend_title,
        )
        if single_figure:
            legend_kw.update(loc="upper right")
            if ss.is_first_row() and ss.is_first_col():
                ax.legend(**legend_kw)
            else:
                ax.legend([], [], **legend_kw)
        else:
            ax.legend(loc="best", **legend_kw)

        # add plot decorations
        if ss.is_last_row():
            ax.set_xlabel("n")
        if ss.is_first_col():
            ax.set_ylabel("RMSE")
        ax.grid(linestyle="--")
        ax.grid(which="minor", linestyle=":")

    # if single_figure:
    #     axs[0].set_ylim(0.95, 1.95)

    return figs


def save_figures(figs: list[plt.Figure]) -> None:
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
    return parser.parse_args(argv)


def main(argv: Sequence[str] = argv[1:]) -> None:
    """Entry point of the script."""
    args = parse_args(argv)
    results = load_results()
    df = results_to_df(results)
    figs = plot(df, args.single_figure)
    save_figures(figs)
    plt.show()


if __name__ == "__main__":
    main()
