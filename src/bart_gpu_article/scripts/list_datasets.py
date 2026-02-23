"""Get a list of OPENML datasets to use."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import polars as pl
from openml.datasets import list_datasets


def get_data() -> pd.DataFrame:
    """Download list of OpenML datasets."""
    return list_datasets(output_format="dataframe")


def filter_data(df: pd.DataFrame) -> pl.DataFrame:
    """Filter list of datasets."""
    return (
        pl.DataFrame(df)
        .rename(
            {
                "NumberOfInstances": "n",
                "NumberOfFeatures": "p",
                "NumberOfClasses": "k",
                "NumberOfNumericFeatures": "pcont",
                "NumberOfSymbolicFeatures": "pcat",
            }
        )
        .group_by("name")
        .agg(
            # keep only the latest version of each dataset
            pl.all().max_by("version")
        )
        .filter(
            # synthetic datasets of which I can not infer curation
            pl.col("name")
            .is_in(
                [
                    "Agrawal1",
                    "friedman1",
                    "Census-(Augmented)",
                    "autos",
                    "breast",
                    "chen_10",
                    "chen_10_null",
                    "colon",
                    "prostate",
                    "Avocado-Prices-(Augmented)",
                    "Chaos_detection_in_Duffing_system",
                    "Credit_Card_Fraud_",
                    "flycraft-demonstrations",
                ]
            )
            .not_(),
            pl.col("name").str.starts_with("BNG(").not_(),
            pl.col("name").str.starts_with("simulated_").not_(),
            pl.col("name").str.starts_with("Hyperplane_").not_(),
            pl.col("name").str.starts_with("SEA(").not_(),
            pl.col("name").str.starts_with("Stagger").not_(),
            pl.col("name").str.starts_with("bates_").not_(),
            # end synthetic datasets
            # datasets with a format which is not convenient for this kind of modeling
            pl.col("name")
            .is_in(
                [
                    "1M-python-questions-on-stackoverflow",
                    "Amazon---Ratings-(Beauty-Products)",
                    "Chess-Position--Chess-Moves",
                    "DBpedia(YAGO).arff",
                    "Edge_Embedding",
                    "Node_Embedding",
                    "Wikidata",
                ]
            )
            .not_(),
            # end wrong format
            # other ignored datasets
            pl.col("name").ne("click"),  # don't know what this is
            pl.col("name").ne("Dominick"),  # too big
            pl.col("name").ne("M4-competition-monthly"),  # too big
            pl.col("name").ne("M4-competition-quarterly"),  # too big
            pl.col("name").ne("NSE-Future-and-Options-Dataset-3M"),  # dunno
            # end other ignored
            # redundant stuff, there's already a better version in the bunch
            pl.col("name").ne("subset_higgs"),  # dunno and seems redundant with Higgs
            pl.col("name").str.starts_with("BAF_variant").not_(),  # a bit redundant
            pl.col("name").ne("Airlines_DepDelay_1M"),  # redundant with 10M version
            pl.col("name").ne("bot-iot-all-features"),  # redundant with BOT-IoT
            pl.col("name").ne("MTPL_SHAP_Tutorial"),  # the OG is freMTPL2freq
            # end redundant stuff
            pl.col("k").eq(0) | pl.col("k").eq(2) | pl.col("k").is_null(),
            # continuous or binary outcome
            (pl.col("n") >= 1_000_000) | (pl.col("did") == 41214),
            # large sample size, but for a single dataset we want (freMTPL2freq)
            NumberOfMissingValues=0,  # no missing values
        )
        .with_columns((pl.col("pcont") / pl.col("p")).alias("pcont_over_p"))
        .sort("name")
    )


def print_data_info(df: pl.DataFrame) -> None:
    """Print the list of names of the selected datasets."""
    for row in df.select(["name", "did"]).iter_rows():
        name, did = row
        print(f"{did:5}  {name}")


def save_dataset_metadata(df: pl.DataFrame) -> None:
    """Save list of dataset ids & info."""
    df = df.select("did", "name")
    file = Path("./data") / "list-datasets.csv"
    file.parent.mkdir(parents=True, exist_ok=True)
    print(f"write {file}...")
    df.write_csv(file)


def plot_datasets_metadata(df: pl.DataFrame) -> plt.Figure:
    """Plot histograms of metadata of datasets."""
    fig, axes = plt.subplots(
        2, 3, figsize=(12, 8), layout="constrained", clear=True, num="list-datasets"
    )
    axes = axes.flatten()

    # Define variables to plot
    hist_vars = [
        "n",
        "p",
        "k",
        "pcont",
        "pcat",
        "pcont_over_p",
    ]

    for ax, col in zip(axes, hist_vars):
        data = df[col]
        assert data.null_count() == 0 or col == "k", col
        data = data.drop_nulls()
        ax.hist(data, bins="auto", histtype="stepfilled")
        ax.set(xlabel=col, ylabel="Count per bin")

    return fig


def save_fig(fig: plt.Figure) -> None:
    """Save a figure."""
    file = Path("./plots") / f"{fig.get_label()}.pdf"
    file.parent.mkdir(parents=True, exist_ok=True)
    print(f"write {file}...")
    fig.savefig(file)


def main() -> None:
    """Entry point of the script."""
    df = get_data()
    df = filter_data(df)
    print_data_info(df)
    save_dataset_metadata(df)
    fig = plot_datasets_metadata(df)
    save_fig(fig)
    plt.show()


if __name__ == "__main__":
    main()
