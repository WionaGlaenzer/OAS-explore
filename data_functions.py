import pandas as pd
import sys
import os
from os.path import basename
import logging

def select_files(filters, input_file="assets/OAS_overview.csv", output_file="outputs/data_to_download.csv"):
    """Filters the dataset based on the given criteria and writes download links to a file."""
    
    logging.info(f"Applying filters: {filters}")

    # Load dataset
    df = pd.read_csv(input_file)

    # Print filters for debugging
    print(f"Applying filters: {filters}")
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
    with open(output_file, 'w') as f:
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

import pandas as pd
import ast


# Function to extract keys as a list, accounting for insertions
def extract_keys_with_insertions(parsed_dict):
    """
    From the parsed ANARCI numbering dictionary, extract all keys (including insertions) as a sorted list.
    """
    if isinstance(parsed_dict, dict):  # Ensure the input is a dictionary
        keys = []
        for region in parsed_dict.values():
            if isinstance(region, dict):
                for key in region.keys():
                    try:
                        # Try to parse the base number (before any letter)
                        base_number = int("".join(filter(str.isdigit, key.strip())))
                        keys.append(base_number)
                    except ValueError:
                        continue  # Skip invalid keys if parsing fails
        return sorted(keys)  # Sort keys for consistency
    return None  # Return None for invalid inputs

def clean_and_parse_dict(column_value):
    """
    Function to clean the anarci numbering string (as saved in OAS) and parse it into a dictionary.
    """
    if pd.notnull(column_value):  # Check for non-null values
        try:
            # Use ast.literal_eval to parse the string safely
            parsed_dict = ast.literal_eval(column_value.strip())
            return parsed_dict  # Return the parsed dictionary
        except (SyntaxError, ValueError):
            # If parsing fails, return None
            return None
    return None

def process_anarci_column(csv_file, output_file):
    """
    Process the ANARCI column in the CSV file and save the result to a new file.
    """
    # Read the CSV file
    df = pd.read_csv(csv_file)
    # Clean and parse the ANARCI column
    df["anarci_list"] = df["ANARCI_numbering"].apply(clean_and_parse_dict)
    df["anarci_list"] = df["anarci_list"].apply(extract_keys_with_insertions)
    # Remove the ANARCI column
    df = df.drop(columns=["ANARCI_numbering"])
    # Do filerting to remove sequences with empty columns
    df = df.dropna(subset=["fwr1_aa", "cdr1_aa", "fwr2_aa", "cdr2_aa", "fwr3_aa", "cdr3_aa", "fwr4_aa"])
    # Define length ranges for each column
    length_ranges = {
        "fwr1_aa": (20, 200),
        "cdr1_aa": (5, 12),
        "cdr2_aa": (1, 10),
        "cdr3_aa": (5, 38),
        "fwr4_aa": (10, 200)
    }
    # Filter rows based on length of entries in each specified column
    for column, (min_length, max_length) in length_ranges.items():
        df = df[df[column].apply(lambda x: min_length <= len(x) <= max_length)]
    # Save the result to a new file
    df.to_csv(output_file, index=False)

def get_sequences_per_individual(oas_overview, sequences):
    """
    Get the number of sequences per individual from the OAS overview CSV file and the sequences CSV file.
    """
    # Read the OAS overview CSV file
    oas_overview = pd.read_csv(oas_overview)
    # Read the sequences CSV file
    sequences = pd.read_csv(sequences)
    # get all unique files from the sequences CSV file
    indices = sequences["File_ID"]
    oas_overview = oas_overview[oas_overview["File_index"].isin(indices)]
    unique_individuals = oas_overview["Subject"].unique()
    # make dictionary with subject name as key and the file_indexes as a list as value
    files_per_individual = {individual: oas_overview[oas_overview["Subject"] == individual]["File_index"].unique().tolist() for individual in unique_individuals}
    # count number of rows in sequences file with the file_indexes in the files_per_individual dictionary
    for individual, files in files_per_individual.items():
        for file in files:
            count = sequences[sequences["File_ID"] == file].shape[0]
            print(f"{individual}: {file} - {count}")
    
    print(files_per_individual)

    return files_per_individual

def separate_individuals(sequences, files_per_individual, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    created_files = []  # List to store created file names
    sequences = pd.read_csv(sequences)

    for i in files_per_individual:

        new_file = f"{output_folder}/individual_{i}.csv"
        
        # Check if file exists to control header writing
        file_exists = os.path.isfile(new_file)


        for file in files_per_individual[i]:
            subset = sequences[sequences["File_ID"] == file]
            
            # Append mode ('a'), write header only if file does not exist
            subset.to_csv(new_file, mode='a', index=False, header=not file_exists)

            # Update file_exists to avoid writing header again in the next iterations
            file_exists = True
        
        created_files.append(new_file)

    created_files_path = f"{output_folder}/created_files.txt"
    with open(created_files_path, "w") as f:
        for file in created_files:
            f.write(file + "\n")

def get_sequences_per_publication(oas_overview, sequences):
    """
    Get the number of sequences per individual from the OAS overview CSV file and the sequences CSV file.
    """
    # Read the OAS overview CSV file
    oas_overview = pd.read_csv(oas_overview)
    # Read the sequences CSV file
    sequences = pd.read_csv(sequences)
    # get all unique files from the sequences CSV file
    indices = sequences["File_ID"]
    oas_overview = oas_overview[oas_overview["File_index"].isin(indices)]
    unique_publications = oas_overview["Author"].unique()
    # make dictionary with subject name as key and the file_indexes as a list as value
    files_per_publication = {publication: oas_overview[oas_overview["Author"] == publication]["File_index"].unique().tolist() for publication in unique_publications}
    # count number of rows in sequences file with the file_indexes in the files_per_individual dictionary
    no_seqs_per_publication = {}
    for publication, files in files_per_publication.items():
        sequences_for_that_publication = 0
        for file in files:
            count = sequences[sequences["File_ID"] == file].shape[0]
            sequences_for_that_publication += count
            #print(f"{publication}: {file} - {count}")
        logging.info(f"Number of sequences from publication {publication}: {count}")
        no_seqs_per_publication[publication] = sequences_for_that_publication
    
    #print(files_per_publication)

    return files_per_publication, no_seqs_per_publication

def separate_publications(sequences, files_per_publication, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    created_files = []  # List to store created file names
    sequences = pd.read_csv(sequences)

    for i in files_per_publication:
        #remove spaces and special characters from i
        i_filename = i.replace(" ", "_")
        i_filename = i_filename.replace(",", "")
        i_filename = i_filename.replace(":", "")
        i_filename = i_filename.replace(";", "")
        i_filename = i_filename.replace(".", "")

        new_file = f"{output_folder}/publication_{i_filename}.csv"
        
        # Check if file exists to control header writing
        file_exists = os.path.isfile(new_file)


        for file in files_per_publication[i]:
            subset = sequences[sequences["File_ID"] == file]
            
            # Append mode ('a'), write header only if file does not exist
            subset.to_csv(new_file, mode='a', index=False, header=not file_exists)

            # Update file_exists to avoid writing header again in the next iterations
            file_exists = True
        
        created_files.append(new_file)

    created_files_path = f"{output_folder}/created_files.txt"
    with open(created_files_path, "w") as f:
        for file in created_files:
            f.write(file + "\n")

def number_of_seqs_overview(input_files, output_file):
    filenames = []
    sequence_numbers = []
    # iterate through the files and save the number of sequences in each
    for file in input_files:
        df = pd.read_csv(file)
        no_of_seqs = len(df)
        filenames.append(basename(file))
        sequence_numbers.append(no_of_seqs)
    # create a dataframe containing filenames and number of sequences
    df_overview = pd.DataFrame({
        "Filename": filenames,
        "Number_of_sequences": sequence_numbers
    })
    # sort dataframe in descending order by Number_of_sequences
    df_overview = df_overview.sort_values(by="Number_of_sequences", ascending=False)
    df_overview.to_csv(output_file, index=False)

def csv_to_txt(input_file, output_file):
    """
    Converts a CSV file to a TXT file by concatenating specified columns from each row.
    
    Parameters:
    - input_file: Path to the input CSV file.
    - output_file: Path to the output TXT file.
    """
    columns = [3,4,5,6,7,8,9]
    with open(input_file, 'r') as in_file, open(output_file, 'w') as out_file:
        # Skip the header line
        next(in_file)
        
        # Read each subsequent line
        for line in in_file:
            # Split the line into columns
            cols = line.strip().split(',')
            # Concatenate the specified columns
            concatenated = ''.join([cols[i] for i in columns])  # Concatenate specified columns
            # Write the concatenated result to the output TXT file
            out_file.write(concatenated + '\n')

