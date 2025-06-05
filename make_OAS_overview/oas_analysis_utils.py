import dask.dataframe as dd

__all__ = ["analyze_isotype_sequences"]


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
