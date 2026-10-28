"""Make a LaTeX table from results saved by the `fullbench` script."""

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace
from collections.abc import Sequence
from math import floor, isfinite, log10
from pathlib import Path
from sys import argv

import polars as pl

from bart_gpu_article.scripts.fullbench_plot import (
    METHOD_NICKNAMES,
    OUTCOME_NICKNAMES,
    dataset_name,
    load_results,
)

METHOD_ORDER = ("xgboost", "bartz", "bartz2000")
"""Order of the method column groups; unknown methods go last."""

MCMC_COLUMNS = ("rmse", "time_train", "tree_life")
"""Columns shown for MCMC methods; the others get only the first two."""

# the table is tight on width, so shorten the longest dataset names
DATASET_NICKNAMES = {
    "australian electricity demand": "electricity",
    "radar traffic data": "radar traffic",
    "delays zurich transport": "zurich delays",
}

# the unit of time is seconds, the caption must say it
COLUMN_HEADERS = {"rmse": "RMSE", "time_train": "time", "tree_life": "life"}

TIME_DECIMALS = {"xgboost": 1, "bartz": 0, "bartz2000": 0}
"""Decimals of the training time per method (default 1). The time is shown
without its spread across rounds, which is negligible, to save width."""


def aggregate(df: pl.DataFrame) -> pl.DataFrame:
    """Mean and sdev across rounds per (method, dataset)."""
    # show the tree life as a fraction of its maximum
    df = df.with_columns(
        pl.col("tree_life").cast(pl.Float64) / pl.col("mcmc_steps").cast(pl.Float64)
    )
    stats = [
        expr
        for col in MCMC_COLUMNS
        for expr in (
            # all-null columns come out of the concat with the null dtype
            pl.col(col).cast(pl.Float64).mean().alias(f"mean_{col}"),
            pl.col(col).cast(pl.Float64).std().alias(f"{col}_sdev"),
        )
    ]
    return df.group_by("method", "dataset_path", maintain_order=True).agg(
        *stats,
        n=pl.col("n_train").drop_nulls().first(),
        p=pl.col("p").drop_nulls().first(),
        # failed runs are recorded as all-None rows, so skip the nulls
        outcome_type=pl.col("outcome_type").drop_nulls().first(),
        is_mcmc=pl.col("move_acc").is_not_null().any(),
    )


def format_uncertain(mean: float | None, sdev: float | None) -> str:
    """Format as `mean(sdev)` with one significant digit on `sdev`."""
    if mean is None:
        return "---"
    if sdev is None or not isfinite(sdev) or sdev == 0:
        # 3 significant digits, but never in exponential notation
        decimals = max(0, 2 - floor(log10(abs(mean)))) if mean else 0
        return f"\\num{{{mean:.{decimals}f}}}"
    decimals = max(0, -floor(log10(sdev)))
    sdev_digits = round(sdev * 10**decimals)
    if sdev_digits == 10:  # rounding carried over to the next digit
        decimals = max(0, decimals - 1)
        sdev_digits = round(sdev * 10**decimals)
    return f"\\num{{{mean:.{decimals}f}({sdev_digits})}}"


def make_table(agg: pl.DataFrame) -> str:
    """Render the aggregated results as a LaTeX tabular."""
    methods = agg.get_column("method").unique(maintain_order=True).to_list()
    methods.sort(key=lambda m: METHOD_ORDER.index(m) if m in METHOD_ORDER else 99)
    is_mcmc = dict(agg.group_by("method").agg(pl.col("is_mcmc").any()).iter_rows())
    columns = {m: MCMC_COLUMNS if is_mcmc[m] else MCMC_COLUMNS[:2] for m in methods}
    rows = {
        (m, d): r
        for m, d, r in zip(
            agg.get_column("method"),
            agg.get_column("dataset_path"),
            agg.iter_rows(named=True),
        )
    }

    # no math in the headers, so that they follow the font of the table
    info_cols = ("dataset", "\\textit{n}", "\\textit{p}", "task")
    spec = "lrrl" + "".join("r" * len(columns[m]) for m in methods)
    lines = [f"\\begin{{tabular}}{{{spec}}}", "\\toprule"]

    group_header = [""] * len(info_cols)
    rules = []
    start = len(info_cols) + 1
    for m in methods:
        ncol = len(columns[m])
        name = METHOD_NICKNAMES.get(m, m)
        group_header.append(f"\\multicolumn{{{ncol}}}{{c}}{{{name}}}")
        rules.append(f"\\cmidrule(lr){{{start}-{start + ncol - 1}}}")
        start += ncol
    lines.append(" & ".join(group_header) + " \\\\")
    lines.append(" ".join(rules))
    header = list(info_cols) + [COLUMN_HEADERS[c] for m in methods for c in columns[m]]
    lines += [" & ".join(header) + " \\\\", "\\midrule"]

    datasets = agg.unique("dataset_path", maintain_order=True)
    for info in datasets.iter_rows(named=True):
        path = info["dataset_path"]
        cells = [
            DATASET_NICKNAMES.get(name := dataset_name(path, "simulated"), name),
            f"\\num{{{info['n']}}}",
            f"\\num{{{info['p']}}}",
            OUTCOME_NICKNAMES[info["outcome_type"]],
        ]
        for m in methods:
            row = rows.get((m, path))
            for c in columns[m]:
                mean = None if row is None else row[f"mean_{c}"]
                if mean is None:
                    cells.append("---")
                elif c == "time_train":
                    cells.append(f"\\num{{{mean:.{TIME_DECIMALS.get(m, 1)}f}}}")
                elif c == "tree_life":
                    cells.append(f"\\num{{{mean:.2f}}}")
                else:
                    cells.append(format_uncertain(mean, row[f"{c}_sdev"]))
        lines.append(" & ".join(cells) + " \\\\")

    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def describe(df: pl.DataFrame) -> None:
    """Print the facts the caption needs to state."""
    for col in ("device", "n_test", "mcmc_steps"):
        values = df.get_column(col).drop_nulls().unique().to_list()
        print(f"{col}: {values}")
    rounds = df.group_by("method", "dataset_path").len().get_column("len").unique()
    print(f"rounds: {rounds.to_list()}")


def save_table(table: str) -> None:
    """Save the table under ./plots."""
    outdir = Path("./plots")
    outdir.mkdir(exist_ok=True)
    file = outdir / "fullbench-table.tex"
    print(f"write {file}...")
    file.write_text(table)


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
    # files from before tree_life was recorded
    for col in ("tree_life", "mcmc_steps"):
        if col not in df.columns:
            df = df.with_columns(pl.lit(None, pl.Float64).alias(col))
    describe(df)
    table = make_table(aggregate(df))
    print(table)
    save_table(table)


if __name__ == "__main__":
    main()
