import dask.dataframe as dd
import argparse
import os # Import os for path manipulation

def calculate_v_call_stats(csv_path):
    """
    Calculates the frequency of V-gene families from the v_call column in a CSV file using Dask.

    Args:
        csv_path (str): The path to the input CSV file.

    Returns:
        dask.dataframe.Series: A Dask Series containing the counts of each V-gene family.
    """
    # Read the CSV file using Dask
    ddf = dd.read_csv(csv_path, dtype={'v_call': 'object', 'd_call': 'object', 'j_call': 'object', 'Redundancy': 'object'})

    # Process the 'v_call' column
    # 1. Extract the part of the string before the first occurrence of '-', '*', or 'S' using regex
    v_families = ddf['v_call'].str.extract(r'^([^-*S]*)', expand=False)

    # Count the occurrences of each unique value
    family_counts = v_families.value_counts()

    # Calculate percentages
    total_count = family_counts.sum()
    family_percentages = (family_counts / total_count * 100)

    # Combine counts and percentages into a single DataFrame
    # Rename the series for clarity before merging
    family_counts = family_counts.rename("Frequency")
    family_percentages = family_percentages.rename("Percentage")

    # Use dd.concat for merging Series along columns (axis=1)
    stats_df = dd.concat([family_counts, family_percentages], axis=1)

    return stats_df # Return the combined DataFrame

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate V-gene family statistics from a CSV file.")
    parser.add_argument("--csv_file", help="Path to the input CSV file.", required=True)
    parser.add_argument("-o", "--output_file", help="Path to save the statistics CSV file.", default=None)
    args = parser.parse_args()

    print(f"Calculating statistics for: {args.csv_file}")

    # Calculate the statistics
    v_call_stats = calculate_v_call_stats(args.csv_file)

    # Compute the results
    computed_stats = v_call_stats.compute()

    # Print unique families
    print("\nUnique V-Gene Families Found:")
    print(computed_stats.index.tolist()) # Get index from computed dataframe

    # Print the results
    print("\nV-Gene Family Statistics (Frequency and Percentage):")
    print(computed_stats)

    # Determine output file path
    if args.output_file:
        output_path = args.output_file
    else:
        # Default output path based on input file name
        base_name = os.path.basename(args.csv_file)
        name, _ = os.path.splitext(base_name)
        output_path = f"{name}_v_call_stats.csv"

    # Save the computed statistics to a CSV file
    print(f"\nSaving statistics to: {output_path}")
    # Save the computed pandas DataFrame
    computed_stats.to_csv(output_path, index_label="V_Gene_Family")

    print("Done.")