import pandas as pd

def select_files(filters, input_file="OAS_overview.csv", output_file="data_to_download.txt"):
    """Filters the dataset based on the given criteria and writes download links to a file."""
    
    # Load dataset
    df = pd.read_csv(input_file)

    # Print filters for debugging
    print(f"Applying filters: {filters}")
    print(df["Disease"].unique())
    # Apply categorical filters
    for key, values in filters.items():
        if isinstance(values, list):  # If the filter is a list (e.g., species, isotype)
            df = df[df[key].isin(values)]
            #print(f"after list filerts: {df}")
        elif isinstance(values, dict):  # If the filter is a min/max range
            if "min" in values:
                df = df[df[key] >= values["min"]]
            if "max" in values:
                df = df[df[key] <= values["max"]]

    # Handle empty results
    if df.empty:
        print("Warning: No matching records found!")
    
    # Write filtered download links to output file
    with open(f"outputs/{output_file}", 'w') as f:
        for _, row in df.iterrows():
            f.write(row['Download_Link'] + '\n')

    print(f"Saved {len(df)} download links to 'outputs/{output_file}'")