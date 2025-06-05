import dask.dataframe as dd
import argparse
import sys
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find and save rows where v_call starts with 'IGHJ' from a large CSV.")
    parser.add_argument("csv_file", help="Path to the input CSV file (e.g., assets/OAS_overview.csv).")
    parser.add_argument("--v_call_col", default="v_call", help="Name of the V-call column (default: v_call).")
    parser.add_argument("--file_id_col", default="File_ID", help="Name of the File ID column (default: File_ID).")
    parser.add_argument("--output_csv", default="ighj_entries.csv", help="Path to the output CSV file (default: ighj_entries.csv).")

    args = parser.parse_args()

    print(f"Reading CSV: {args.csv_file}")
    try:
        # Specify dtype for potentially mixed-type columns if known, otherwise let Dask infer
        # Using low_memory=False might help with mixed types but uses more memory
        ddf = dd.read_csv(args.csv_file, low_memory=False)
        print("CSV read successfully.")
    except Exception as e:
        print(f"Error reading CSV file: {e}", file=sys.stderr)
        sys.exit(1)

    # Verify required columns exist
    required_cols = [args.v_call_col, args.file_id_col]
    missing_cols = [col for col in required_cols if col not in ddf.columns]
    if missing_cols:
        print(f"Error: Missing required columns: {', '.join(missing_cols)}", file=sys.stderr)
        print(f"Available columns: {list(ddf.columns)}", file=sys.stderr)
        sys.exit(1)

    print(f"Filtering rows where '{args.v_call_col}' starts with 'IGHJ'...")
    # Ensure the v_call column is treated as string before filtering
    ddf[args.v_call_col] = ddf[args.v_call_col].astype(str)
    ighj_rows = ddf[ddf[args.v_call_col].str.startswith('IGHJ', na=False)]

    print(f"Saving filtered rows to: {args.output_csv}")
    try:
        # Use Dask's to_csv for potentially large results, saving as a single file
        # Compute the number of rows first to check if it's empty
        # len() on a Dask DataFrame/Series/Index triggers computation and returns an int
        count = len(ighj_rows)
        if count == 0:
             print("No rows found where v_call starts with 'IGHJ'. No output file created.")
        else:
            print(f"Found {count} rows where v_call starts with 'IGHJ'. Saving...")
            # Ensure the output directory exists if specified in the path
            output_dir = os.path.dirname(args.output_csv)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
                print(f"Created output directory: {output_dir}")
            ighj_rows.to_csv(args.output_csv, single_file=True, index=False)
            print(f"Filtered rows saved successfully to {args.output_csv}")

    except Exception as e:
        print(f"Error during computation or saving: {e}", file=sys.stderr)
        sys.exit(1)

    print("Script finished.")
