import pandas as pd
import sys

def select_files(filters, input_file="assets/OAS_overview.csv", output_file="data_to_download.csv"):
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
        f.write("File_index,Download_Link\n")
        for _, row in df.iterrows():
            f.write(f"{row['File_index']},{row['Download_Link']}\n")

    print(f"Saved {len(df)} download links to 'outputs/{output_file}'")

def csv_to_fasta(input_csv, output_fasta):
    """
    Convert CSV file to FASTA format.
    Uses the Sequence_ID from the CSV as the sequence identifier.
    """
    with open(input_csv, 'r') as csv_file, open(output_fasta, 'w') as fasta_file:
        # Skip header
        header = csv_file.readline()
        
        for line in csv_file:
            fields = line.strip().split(',')
            sequence_id = f"Sequence_{fields[0]}"  # Sequence_ID is now the first column
            sequence = fields[3] + fields[4] + fields[5] + fields[6] + fields[7] + fields[8] + fields[9]
            
            # Write in FASTA format
            fasta_file.write(f">{sequence_id}\n{sequence}\n")

    print(f"FASTA file created successfully at {output_fasta}")

def filter_representative_sequences(fasta_file, input_csv, output_csv):
    """
    Create a new CSV file containing only sequences that are in the representative FASTA file.
    Handles FASTA headers in format '>Sequence_225'
    """
    # Read sequence IDs from FASTA file
    representative_ids = set()
    with open(fasta_file, 'r') as f:
        for line in f:
            if line.startswith('>'):
                # Extract the number after 'Sequence_'
                seq_id = line.strip().split('_')[1]
                representative_ids.add(seq_id)
    
    # Filter CSV file to keep only representative sequences
    with open(input_csv, 'r') as in_f, open(output_csv, 'w') as out_f:
        # First line is header - write it directly
        header = next(in_f)
        out_f.write(header)
        
        # Process remaining lines
        for line in in_f:
            seq_id = line.split(',')[0]  # Get the ID from first column
            if seq_id in representative_ids:
                out_f.write(line)