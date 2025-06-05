"""Utility script to compute isotype sequence statistics using Dask."""

from pathlib import Path
import argparse
import dask.dataframe as dd

from oas_analysis_utils import analyze_isotype_sequences


def main() -> None:
    """Run the command line interface for the isotype analysis."""
    parser = argparse.ArgumentParser(
        description=(
            "Analyze sequence counts per isotype from a CSV file using Dask, "
            "separating by species."
        )
    )
    parser.add_argument("csv_file", help="Path to the input CSV file.")
    parser.add_argument(
        "-o",
        "--output-dir",
        default=".",
        help="Directory in which result CSVs will be written.",
    )
    args = parser.parse_args()

    print(f"Starting analysis for: {args.csv_file}")

    print(f"Reading CSV: {args.csv_file}")
    ddf = dd.read_csv(args.csv_file)

    # Calculate and print the number of unique authors
    print("Calculating number of unique authors...")
    unique_author_count = ddf['Author'].nunique().compute()
    print(f"\nNumber of unique authors found: {unique_author_count}")

    # Define species lists
    mouse_strains = [
        "mouse_C57BL/6",
        "mouse_BALB/c",
        "mouse",
        "Kymouse",
        "HIS-Mouse",
        "mouse_RAG2-GFP/129Sve",
        "mouse_Swiss-Webster",
    ]
    human_species = "human"

    # --- Analysis for all species ---
    print("--- Analyzing all species ---")
    all_species_stats = analyze_isotype_sequences(ddf)
    print("Computing results for all species...")
    computed_all_stats = all_species_stats.compute()
    print("Aggregated Sequence Counts and Percentages per Isotype (All Species):")
    print(computed_all_stats)
    output_dir = Path(args.output_dir)

    output_path_all = output_dir / "isotype_statistics_all_species.csv"
    print(f"Saving all species statistics to: {output_path_all}")
    computed_all_stats.to_csv(output_path_all, index=False)

    # --- Analysis for human species ---
    print("--- Analyzing human species ---")
    human_stats = analyze_isotype_sequences(ddf, species_filter=human_species)
    print("Computing results for human species...")
    computed_human_stats = human_stats.compute()
    print("Aggregated Sequence Counts and Percentages per Isotype (Human):")
    print(computed_human_stats)
    output_path_human = output_dir / "isotype_statistics_human.csv"
    print(f"Saving human statistics to: {output_path_human}")
    computed_human_stats.to_csv(output_path_human, index=False)

    # --- Analysis for mouse species ---
    print("--- Analyzing mouse species ---")
    mouse_stats = analyze_isotype_sequences(ddf, species_filter=mouse_strains)
    print("Computing results for mouse species...")
    computed_mouse_stats = mouse_stats.compute()
    print("Aggregated Sequence Counts and Percentages per Isotype (Mouse):")
    print(computed_mouse_stats)
    output_path_mouse = output_dir / "isotype_statistics_mouse.csv"
    print(f"Saving mouse statistics to: {output_path_mouse}")
    computed_mouse_stats.to_csv(output_path_mouse, index=False)


    print("Analysis complete. Results saved for all species, human, and mouse.")


if __name__ == "__main__":
    main()

