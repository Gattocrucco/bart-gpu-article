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


def make_figures(
    num_groups: int, single_figure: bool, column: str
) -> tuple[list[plt.Figure], list[plt.Axes]]:
    """Create the figures and axes for the plotting functions."""
    # reset matplotlib
    plt.close("all")
    plt.rcdefaults()

    if single_figure:
        fig, axs = plt.subplots(
            2,
            2,
            figsize=[8.5, 8.5],
            num=f"compare-bart-packages-plot-{column}",
            clear=True,
            layout="constrained",
            sharex=True,
            sharey=True,
        )
        axs = list(axs.flat)
        figs = [fig]
    else:
        axs = []
        figs = []
        for i in range(num_groups):
            fig, ax = plt.subplots(
                figsize=[4.5, 3.5],
                num=f"compare-bart-packages-plot-{column}-{i}",
                clear=True,
                layout="constrained",
            )
            axs.append(ax)
            figs.append(fig)
    axs[1], axs[3] = axs[3], axs[1]

    return figs, axs


def make_groups(df: pl.DataFrame) -> dict[tuple, pl.DataFrame]:
    """Split the dataframe by configuration, keyed by (name, value) pairs."""
    keynames = ["n/ntree", "ntree", "n/p", "p"]
    return {
        tuple(zip(keynames, keys)): group
        for keys, group in df.group_by(keynames, maintain_order=True)
    }


def plot_mse(
    groups: dict[tuple, pl.DataFrame], single_figure: bool, column: str
) -> list[plt.Figure]:
    figs, axs = make_figures(len(groups), single_figure, column)

    cycler = get_cycler()
    all_n = pl.concat(list(groups.values()))["n"]

    for ax, (keys, group) in zip(axs, groups.items()):
        # plot mse curves
        ax.set_prop_cycle(cycler)
        for (package,), data in group.group_by(["package"], maintain_order=True):
            ax.plot(data["n"], data[column], markerfacecolor="none", label=package)

        # prepare data to plot variance references
        var_labels = {
            "eps_var": "error var",
            "pop_var": "population var",
            "prior_var": "prior var",
        }
        vd_check = group.group_by("n").agg(pl.col(*var_labels).n_unique())
        assert vd_check.drop("n").select(pl.all_horizontal(pl.all() == 1).all()).item()
        vd = (
            group.group_by("n")
            .agg(
                pl.col(*var_labels).first(),
            )
            .sort("n")
        )

        # plot variance references
        error_lines = []
        for key, label in var_labels.items():
            (line,) = ax.plot(vd["n"], vd[key], "--k", label=label)
            error_lines.append(line)

        # set plot properties that better be set before plotting labels
        ax.set_xscale("log")
        ref_n = all_n if single_figure else vd["n"]
        ax.set_xlim(
            10 ** math.floor(math.log10(ref_n.min())),
            10 ** math.ceil(math.log10(ref_n.max())),
        )

        # add labels on top of variance lines
        labelLines(
            error_lines,
            xvals=[1000, 2000, 300],
            drop_label=True,
            outline_width=6,
            align=False,
        )

        # add legend; possibly abuse the legend as generic box with text
        ss = ax.get_subplotspec()
        legend_title = "\n".join(
            f"{name}={value}" for name, value in keys if value is not None
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
            ax.set_ylabel("MSE")
        ax.grid(linestyle="--")
        ax.grid(which="minor", linestyle=":")

    # if single_figure:
    #     axs[0].set_ylim(0.95**2, 1.95**2)

    return figs


def plot_generic(
    groups: dict[tuple, pl.DataFrame], single_figure: bool, column: str
) -> list[plt.Figure]:
    figs, axs = make_figures(len(groups), single_figure, column)

    cycler = get_cycler()
    all_n = pl.concat(list(groups.values()))["n"]

    for ax, (keys, group) in zip(axs, groups.items()):
        # plot curves
        ax.set_prop_cycle(cycler)
        for (package,), data in group.group_by(["package"], maintain_order=True):
            ax.plot(data["n"], data[column], markerfacecolor="none", label=package)

        # set log scale on n
        ax.set_xscale("log")
        ref_n = all_n if single_figure else group["n"]
        ax.set_xlim(
            10 ** math.floor(math.log10(ref_n.min())),
            10 ** math.ceil(math.log10(ref_n.max())),
        )

        # add legend; possibly abuse the legend as generic box with text
        ss = ax.get_subplotspec()
        legend_title = "\n".join(
            f"{name}={value}" for name, value in keys if value is not None
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
            ax.set_ylabel(column)
        ax.grid(linestyle="--")
        ax.grid(which="minor", linestyle=":")

    return figs


def plot_coverage(
    groups: dict[tuple, pl.DataFrame], single_figure: bool, column: str
) -> list[plt.Figure]:
    figs = plot_generic(groups, single_figure, column)

    level = int(column.rsplit("_", 1)[-1]) / 100
    target = "latent mean" if "truth" in column else "data"
    for fig in figs:
        for ax in fig.axes:
            # draw the target line over the full x range and label it at low n
            xlim = ax.get_xlim()
            (line,) = ax.plot(xlim, [level, level], "--k", label=f"{level:.0%}")
            ax.set_xlim(xlim)
            ax.set_ylim(0, 1)
            xval = 10 ** (0.9 * math.log10(xlim[0]) + 0.1 * math.log10(xlim[1]))
            labelLines(
                [line], xvals=[xval], drop_label=True, outline_width=6, align=False
            )
            if ax.get_ylabel():
                ax.set_ylabel(f"coverage of {level:.0%} intervals on {target}")

    return figs


PLOT_FUNCTIONS = {
    "mse": plot_mse,
    "coverage_50": plot_coverage,
    "coverage_90": plot_coverage,
    "coverage_truth_50": plot_coverage,
    "coverage_truth_90": plot_coverage,
}


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
    parser.add_argument(
        "-w",
        "--what",
        default="mse",
        help="the results column to plot",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] = argv[1:]) -> None:
    """Entry point of the script."""
    args = parse_args(argv)
    results = load_results()
    df = results_to_df(results)
    groups = make_groups(df)
    plot = PLOT_FUNCTIONS.get(args.what, plot_generic)
    figs = plot(groups, args.single_figure, args.what)
    save_figures(figs)
    plt.show()


if __name__ == "__main__":
    main()
