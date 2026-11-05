"""Make LaTeX tables from results saved by the `fullbench` script: a short one
with the datasets as rows, and a full one with the datasets as columns."""

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

METHOD_ORDER = ("xgboost", "bartz", "xgboostes", "bartz2000")
"""Order of the method column groups; unknown methods go last."""

MCMC_COLUMNS = ("rmse", "time_train", "tree_life")
"""Columns shown for MCMC methods in the short table; the others get only the
first two."""

MAX_LEAVES = {"xgboost": 64, "xgboostes": 64, "bartz": 32, "bartz2000": 32}
"""Maximum number of leaves per tree at the default depth limit, the unit of
the leaves in the full table."""

EMPTY_TREES_PRIOR = 0.05
"""Prior probability of an empty tree in bartz, the unit of the fraction of
empty trees in the full table."""

FULL_ROWS = {
    "rmse": "RMSE",
    "error_sdev": "error sdev",
    "relmse": "MSE/test var",
    "logloss": "log loss",
    "coverage_50": "50\\% coverage",
    "time_train": "train time (s)",
    "time_test": "cold predict (s)",
    "time_test_warm": "warm predict (s)",
    "peak_memory": "peak GPU mem.\n(GB)",
    "num_trees": "trees",
    "mean_leaves": "leaves",
    "frac_empty_trees": f"empty trees/{EMPTY_TREES_PRIOR}",
    "move_acc": "acceptance",
    "tree_life": "tree life/{steps}",
}
"""Per-method fields of the full table, in order, with their labels. Methods
that lack a field throughout are left out of its rows. A label with newlines
goes one line per row of its group."""

JIT_METHODS = ("bartz", "bartz2000")
"""Methods whose times include the jax compilation, since each round runs in a
fresh process without a warm-up call."""

JIT_COLUMNS = ("time_train", "time_test")
"""The times that include the compilation, i.e., not the warm predict."""

STAT_COLUMNS = (*FULL_ROWS, "test_sdev")
"""Columns that get a mean and sdev across rounds."""

SIMULATED = {
    "australian electricity demand": False,
    # "The data has been produced using Monte Carlo simulations" (OpenML 45570)
    "higgs": True,
    "radar traffic data": False,
    "delays zurich transport": False,
    # UCI poker hand: random draws from a deck, labeled by the poker rules
    "poker": True,
    # savedata
    "simulated": True,
}
"""Whether each dataset is simulated, keyed on `dataset_name`. From the OpenML
descriptions; a new dataset must be added by hand."""

TIME_SERIES = {
    # half-hourly demand of 5 Australian states
    "australian electricity demand": True,
    "higgs": False,
    # traffic counts from radar sensors in Austin over time
    "radar traffic data": True,
    # delays of the Zurich public transport over a month
    "delays zurich transport": True,
    "poker": False,
    "simulated": False,
}
"""Whether each dataset is a time series, keyed on `dataset_name`. The train/test
split is random, so on these the test points sit between training points in
time."""

OPENML_IDS = {
    "australian electricity demand": 46237,
    "higgs": 45570,
    "radar traffic data": 43354,
    "delays zurich transport": 45045,
    "poker": 43955,
}
"""OpenML ids of the real datasets, keyed on `dataset_name`, as found in the
list written by `list_datasets` that `save_datasets` picks from."""

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
    df = df.with_columns(
        # show the tree life as a fraction of its maximum
        pl.col("tree_life").cast(pl.Float64) / pl.col("mcmc_steps").cast(pl.Float64),
        pl.col("mean_leaves").cast(pl.Float64)
        / pl.col("method").replace_strict(MAX_LEAVES, default=None),
        pl.col("frac_empty_trees").cast(pl.Float64) / EMPTY_TREES_PRIOR,
        # the mse relative to predicting the test mean, as in fullbench-plot
        relmse=(pl.col("rmse") / pl.col("test_sdev")).pow(2),
        # the lower of the two peak memory measures, in GB: the one of jax
        # when there is, since the gpu one also counts what the jax
        # allocator holds without using, and the gpu one otherwise
        peak_memory=pl.min_horizontal(
            pl.col("peak_gpu_memory").cast(pl.Float64),
            pl.col("peak_jax_memory").cast(pl.Float64),
        )
        / 1e9,
    )
    stats = [
        expr
        for col in STAT_COLUMNS
        for expr in (
            # all-null columns come out of the concat with the null dtype
            pl.col(col).cast(pl.Float64).mean().alias(f"mean_{col}"),
            pl.col(col).cast(pl.Float64).std().alias(f"{col}_sdev"),
        )
    ]
    return df.group_by("method", "dataset_path", maintain_order=True).agg(
        *stats,
        n=pl.col("n_train").drop_nulls().first(),
        mcmc_steps=pl.col("mcmc_steps").drop_nulls().first(),
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


def make_full_table(agg: pl.DataFrame) -> str:
    """Render the aggregated results as a LaTeX tabular with the datasets as
    columns and one row per (field, method)."""
    methods = agg.get_column("method").unique(maintain_order=True).to_list()
    methods.sort(key=lambda m: METHOD_ORDER.index(m) if m in METHOD_ORDER else 99)
    rows = {(r["method"], r["dataset_path"]): r for r in agg.iter_rows(named=True)}
    datasets = agg.unique("dataset_path", maintain_order=True).to_dicts()
    paths = [d["dataset_path"] for d in datasets]

    lines = [f"\\begin{{tabular}}{{ll{'r' * len(datasets)}}}", "\\toprule"]
    names = [dataset_name(d["dataset_path"], "simulated") for d in datasets]
    names = [DATASET_NICKNAMES.get(name, name) for name in names]
    # the names are wider than the numbers, so put one word per line
    names = [name.split(" ") for name in names]
    for words in names:
        if words == ["simulated"]:
            words.append("(\\autoref{eq:dgp})")
    names = [
        f"\\begin{{tabular}}[b]{{@{{}}r@{{}}}}{' \\\\ '.join(words)}\\end{{tabular}}"
        if len(words) > 1
        else words[0]
        for words in names
    ]
    lines += [" & ".join(["", "", *names]) + " \\\\", "\\midrule"]

    # per-dataset info, the same for all methods
    ids = [
        OPENML_IDS.get(dataset_name(d["dataset_path"], "simulated")) for d in datasets
    ]
    info = {
        # not \\num, the ids are not quantities to group by thousands
        "OpenML id": [
            "---" if i is None else f"\\href{{https://www.openml.org/d/{i}}}{{{i}}}"
            for i in ids
        ],
        "\\textit{n}": [f"\\num{{{d['n']}}}" for d in datasets],
        "\\textit{p}": [f"\\num{{{d['p']}}}" for d in datasets],
        "task": [OUTCOME_NICKNAMES[d["outcome_type"]] for d in datasets],
        "simulated": [
            "yes" if SIMULATED[dataset_name(d["dataset_path"], "simulated")] else "no"
            for d in datasets
        ],
        "time series": [
            "yes" if TIME_SERIES[dataset_name(d["dataset_path"], "simulated")] else "no"
            for d in datasets
        ],
        # the test set is the same across methods for a given round
        "test sdev": [
            format_uncertain(d["mean_test_sdev"], d["test_sdev_sdev"]) for d in datasets
        ],
    }
    lines += [
        " & ".join([label, "", *cells]) + " \\\\" for label, cells in info.items()
    ]
    lines.append("\\midrule")

    # the tree life is divided by the number of steps, so it must be the same
    steps = agg.get_column("mcmc_steps").drop_nulls().unique().item()

    groups = []
    for col, label in FULL_ROWS.items():
        label = label.format(steps=steps)
        group = []
        for m in methods:
            cells = []
            for path in paths:
                row = rows.get((m, path))
                mean = None if row is None else row[f"mean_{col}"]
                sdev = None if row is None else row[f"{col}_sdev"]
                cells.append(format_uncertain(mean, sdev))
            if any(c != "---" for c in cells):
                group.append((m, cells))
        rows_ = []
        label_lines = label.split("\n")
        for i, (m, cells) in enumerate(group):
            line = label_lines[i] if i < len(label_lines) else ""
            if col == "mean_leaves":
                # the unit differs by method, so put it on each line
                line = (label if i == 0 else f"\\phantom{{{label}}}") + (
                    f"/{MAX_LEAVES[m]}"
                )
            if col in JIT_COLUMNS and m in JIT_METHODS:
                line = f"{line} (incl. XLA)" if line else "(incl. XLA)"
            name = METHOD_NICKNAMES.get(m, m)
            rows_.append(" & ".join([line, name, *cells]) + " \\\\")
        if rows_:
            groups.append(rows_)
    for i, group in enumerate(groups):
        if i:
            lines.append("\\addlinespace")
        lines += group

    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def describe(df: pl.DataFrame) -> None:
    """Print the facts the caption needs to state."""
    for col in ("device", "n_test", "mcmc_steps"):
        values = df.get_column(col).drop_nulls().unique().to_list()
        print(f"{col}: {values}")
    rounds = df.group_by("method", "dataset_path").len().get_column("len").unique()
    print(f"rounds: {rounds.to_list()}")


def save_table(table: str, name: str) -> None:
    """Save the table under ./plots."""
    outdir = Path("./plots")
    outdir.mkdir(exist_ok=True)
    file = outdir / f"{name}.tex"
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
    # files from before these columns were recorded
    for col in (*STAT_COLUMNS, "mcmc_steps"):
        if col not in df.columns:
            df = df.with_columns(pl.lit(None, pl.Float64).alias(col))
    describe(df)
    agg = aggregate(df)
    for make, name in [
        (make_table, "fullbench-table"),
        (make_full_table, "fullbench-table-full"),
    ]:
        table = make(agg)
        print(table)
        save_table(table, name)


if __name__ == "__main__":
    main()
