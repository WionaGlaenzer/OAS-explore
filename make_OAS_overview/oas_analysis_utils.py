import dask.dataframe as dd
import pandas as pd
import matplotlib.pyplot as plt

__all__ = [
    "analyze_isotype_sequences",
    "load_oas_overview",
    "plot_grouped_data",
    "plot_filtered_labels",
]


def analyze_isotype_sequences(ddf, species_filter=None):
    """Analyze sequence counts per isotype.

    Parameters
    ----------
    ddf : dask.dataframe.DataFrame
        Input data frame containing sequence information.
    species_filter : str or list, optional
        Species or list of species to filter on. If ``None`` all
        entries are used.

    Returns
    -------
    dask.dataframe.DataFrame
        Aggregated statistics per isotype with total and percentage
        counts.
    """
    if species_filter:
        if isinstance(species_filter, list):
            ddf_filtered = ddf[ddf["Species"].isin(species_filter)]
        else:
            ddf_filtered = ddf[ddf["Species"] == species_filter]
    else:
        ddf_filtered = ddf

    ddf_filtered["Unique_sequences"] = dd.to_numeric(
        ddf_filtered["Unique_sequences"], errors="coerce"
    ).fillna(0)
    ddf_filtered["Total_sequences"] = dd.to_numeric(
        ddf_filtered["Total_sequences"], errors="coerce"
    ).fillna(0)

    isotype_sums = ddf_filtered.groupby("Isotype").agg(
        {"Unique_sequences": "sum", "Total_sequences": "sum"}
    )

    total_unique = isotype_sums["Unique_sequences"].sum()
    total_total = isotype_sums["Total_sequences"].sum()

    isotype_sums["Percentage_Unique"] = (
        isotype_sums["Unique_sequences"] / total_unique * 100
        if total_unique.compute() > 0
        else 0
    )
    isotype_sums["Percentage_Total"] = (
        isotype_sums["Total_sequences"] / total_total * 100
        if total_total.compute() > 0
        else 0
    )

    return isotype_sums.reset_index().rename(
        columns={
            "Unique_sequences": "Total_Unique_Sequences",
            "Total_sequences": "Total_Total_Sequences",
            "Percentage_Unique": "Percentage_Unique_Sequences",
            "Percentage_Total": "Percentage_Total_Sequences",
        }
    )


def load_oas_overview(path: str) -> dd.DataFrame:
    """Load the OAS overview CSV using Dask and clean numeric columns."""

    ddf = dd.read_csv(path)
    #ddf["Age"] = dd.to_numeric(ddf["Age"], errors="coerce")
    ddf["Unique_sequences"] = dd.to_numeric(ddf["Unique_sequences"], errors="coerce")
    #ddf = ddf[(ddf["Age"].notnull()) & (ddf["Unique_sequences"].notnull())]
    return ddf


def _compute_grouped_sum(df, group_columns, sum_column):
    """Helper to group and sum for both pandas and dask DataFrames."""

    if isinstance(df, dd.DataFrame):
        grouped = df.groupby(group_columns)[sum_column].sum().compute()
    else:
        grouped = df.groupby(group_columns)[sum_column].sum()
    return grouped


def plot_grouped_data(df, group_columns, sum_column, y_label, title, file_name, log_scale=False):
    """Group data by columns, sum, and plot a bar chart."""

    grouped_data = _compute_grouped_sum(df, group_columns, sum_column)

    plt.figure(figsize=(20, 6))
    ax = grouped_data.plot(kind="bar", color="skyblue", edgecolor="black")

    # Set grid below the bars and enable light gray horizontal lines
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="--", color="gray", alpha=0.3)

    # Remove top and right spines (box lines)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if log_scale:
        ax.set_yscale("log")

    plt.xticks(rotation=45, ha="right")
    plt.ylabel(y_label, fontsize=12)
    plt.title(title, fontsize=14)
    plt.tick_params(axis="both", which="major", labelsize=10)
    plt.tight_layout()
    plt.savefig(file_name)
    plt.close()


def plot_filtered_labels(
    df,
    group_columns,
    sum_column,
    y_label,
    title,
    file_name,
    label_threshold,
    log_scale=False,
):
    """Plot a bar chart with labels shown only for entries above a threshold."""

    grouped_data = _compute_grouped_sum(df, group_columns, sum_column)
    labels = [idx if value > label_threshold else "" for idx, value in zip(grouped_data.index, grouped_data.values)]

    plt.figure(figsize=(20, 16))
    bars = plt.bar(range(len(grouped_data)), grouped_data.values, color="skyblue", edgecolor="black")

    plt.xticks(ticks=range(len(labels)), labels=labels, rotation=45, ha="right", fontsize=10)

    if log_scale:
        plt.yscale("log")

    plt.ylabel(y_label, fontsize=14)
    plt.title(title, fontsize=16, pad=20)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tick_params(axis="both", which="major", labelsize=12)

    for bar in bars:
        height = bar.get_height()
        if height > label_threshold:
            plt.text(bar.get_x() + bar.get_width() / 2.0, height, f"{int(height)}", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    plt.savefig(file_name, bbox_inches="tight")
    plt.close()

