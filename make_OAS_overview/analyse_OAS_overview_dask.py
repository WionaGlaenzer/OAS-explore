import dask.dataframe as dd
import argparse
import os

def analyze_isotype_sequences(ddf, species_filter=None):
    """
    Analyzes a Dask DataFrame to calculate the sum and percentage of unique and total sequences per isotype,
    optionally filtering by species.

    Args:
        ddf (dask.dataframe.DataFrame): The input Dask DataFrame.
        species_filter (str or list, optional): Species to filter by. Can be a single string or a list of strings. Defaults to None (no filter).

    Returns:
        dask.dataframe.DataFrame: A Dask DataFrame with the aggregated sums and percentages per isotype.
    """
    # Filter by species if a filter is provided
    if species_filter:
        print(f"Filtering for species: {species_filter}")
        if isinstance(species_filter, list):
            ddf_filtered = ddf[ddf['Species'].isin(species_filter)]
        else:
            ddf_filtered = ddf[ddf['Species'] == species_filter]
    else:
        print("Analyzing all species.")
        ddf_filtered = ddf

    # Ensure sequence columns are numeric, coercing errors to NaN
    ddf_filtered['Unique_sequences'] = dd.to_numeric(ddf_filtered['Unique_sequences'], errors='coerce')
    ddf_filtered['Total_sequences'] = dd.to_numeric(ddf_filtered['Total_sequences'], errors='coerce')

    # Fill potential NaNs resulting from coercion with 0 before summing
    ddf_filtered['Unique_sequences'] = ddf_filtered['Unique_sequences'].fillna(0)
    ddf_filtered['Total_sequences'] = ddf_filtered['Total_sequences'].fillna(0)

    print("Grouping by Isotype and aggregating...")
    # Group by 'Isotype' and aggregate the sums
    isotype_sums = ddf_filtered.groupby('Isotype').agg({
        'Unique_sequences': 'sum',
        'Total_sequences': 'sum'
    })

    # Calculate overall totals for percentage calculation
    total_unique = isotype_sums['Unique_sequences'].sum()
    total_total = isotype_sums['Total_sequences'].sum()

    # Calculate percentages, handle potential division by zero if totals are zero
    isotype_sums['Percentage_Unique'] = (isotype_sums['Unique_sequences'] / total_unique * 100) if total_unique.compute() > 0 else 0
    isotype_sums['Percentage_Total'] = (isotype_sums['Total_sequences'] / total_total * 100) if total_total.compute() > 0 else 0


    # Reset index to make 'Isotype' a column again
    isotype_stats = isotype_sums.reset_index()

    # Rename columns for clarity
    isotype_stats = isotype_stats.rename(columns={
        'Unique_sequences': 'Total_Unique_Sequences',
        'Total_sequences': 'Total_Total_Sequences',
        'Percentage_Unique': 'Percentage_Unique_Sequences',
        'Percentage_Total': 'Percentage_Total_Sequences'
    })

    return isotype_stats

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze sequence counts per isotype from a CSV file using Dask, separating by species.")
    parser.add_argument("csv_file", help="Path to the input CSV file.")
    # Output file argument removed as paths are hardcoded
    args = parser.parse_args()

    print(f"Starting analysis for: {args.csv_file}")

    # Read the CSV file using Dask
    print(f"Reading CSV: {args.csv_file}")
    # Define dtypes for potentially problematic columns or large numbers if necessary
    # Example: dtype={'Age': 'object', 'Unique_sequences': 'float64', 'Total_sequences': 'float64'}
    ddf = dd.read_csv(args.csv_file)

    # Calculate and print the number of unique authors
    print("Calculating number of unique authors...")
    unique_author_count = ddf['Author'].nunique().compute()
    print(f"\nNumber of unique authors found: {unique_author_count}")

    # Define species lists
    mouse_strains = [
        "mouse_C57BL/6", "mouse_BALB/c", "mouse", "Kymouse",
        "HIS-Mouse", "mouse_RAG2-GFP/129Sve", "mouse_Swiss-Webster"
    ]
    human_species = "human"

    # --- Analysis for all species ---
    print("--- Analyzing all species ---")
    all_species_stats = analyze_isotype_sequences(ddf)
    print("Computing results for all species...")
    computed_all_stats = all_species_stats.compute()
    print("Aggregated Sequence Counts and Percentages per Isotype (All Species):")
    print(computed_all_stats)
    output_path_all = "/REDACTED/PATHnzer/Coding/plm_training_pipeline/make_OAS_overview/isotype_statistics_all_species.csv"
    print(f"Saving all species statistics to: {output_path_all}")
    computed_all_stats.to_csv(output_path_all, index=False)

    # --- Analysis for human species ---
    print("--- Analyzing human species ---")
    human_stats = analyze_isotype_sequences(ddf, species_filter=human_species)
    print("Computing results for human species...")
    computed_human_stats = human_stats.compute()
    print("Aggregated Sequence Counts and Percentages per Isotype (Human):")
    print(computed_human_stats)
    output_path_human = "/REDACTED/PATHnzer/Coding/plm_training_pipeline/make_OAS_overview/isotype_statistics_human.csv"
    print(f"Saving human statistics to: {output_path_human}")
    computed_human_stats.to_csv(output_path_human, index=False)

    # --- Analysis for mouse species ---
    print("--- Analyzing mouse species ---")
    mouse_stats = analyze_isotype_sequences(ddf, species_filter=mouse_strains)
    print("Computing results for mouse species...")
    computed_mouse_stats = mouse_stats.compute()
    print("Aggregated Sequence Counts and Percentages per Isotype (Mouse):")
    print(computed_mouse_stats)
    output_path_mouse = "/REDACTED/PATHnzer/Coding/plm_training_pipeline/make_OAS_overview/isotype_statistics_mouse.csv"
    print(f"Saving mouse statistics to: {output_path_mouse}")
    computed_mouse_stats.to_csv(output_path_mouse, index=False)


    print("Analysis complete. Results saved for all species, human, and mouse.")
