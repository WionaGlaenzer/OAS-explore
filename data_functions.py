import pandas as pd
import sys
import os
from os.path import basename, splitext, dirname
import logging
from itertools import cycle
import matplotlib.pyplot as plt
import re
import numpy as np
import pandas as pd
import ast
from datasets import load_dataset, DatasetDict
from transformers import RobertaTokenizer

def select_files(filters, input_file="assets/OAS_overview.csv", output_file="outputs/data_to_download.csv"):
    """
    Filters the dataset based on the criteria given in config.yaml and writes download links to a file.
    """
    
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
    Convert CSV file to FASTA format to be used as input to linclust.
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
    Create a new CSV file containing only sequences that are in the representative FASTA file from linclust.
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

def process_anarci_column_old(csv_file, output_file):
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

def process_anarci_column(csv_file, output_file, chunk_size=1000):
    """
    Process the ANARCI column in the CSV file line by line to reduce memory usage.
    """
    # Define length ranges for filtering
    length_ranges = {
        "fwr1_aa": (20, 200),
        "cdr1_aa": (5, 12),
        "cdr2_aa": (1, 10),
        "cdr3_aa": (5, 38),
        "fwr4_aa": (10, 200)
    }
    
    # Open the output file and write the header first
    with open(output_file, 'w') as f_out:
        first_chunk = True
        
        for chunk in pd.read_csv(csv_file, chunksize=chunk_size):
            # Clean and parse the ANARCI column
            chunk["anarci_list"] = chunk["ANARCI_numbering"].apply(clean_and_parse_dict)
            chunk["anarci_list"] = chunk["anarci_list"].apply(extract_keys_with_insertions)
            
            # Remove the ANARCI column
            chunk = chunk.drop(columns=["ANARCI_numbering"])
            
            # Remove rows with empty required columns
            chunk = chunk.dropna(subset=["fwr1_aa", "cdr1_aa", "fwr2_aa", "cdr2_aa", "fwr3_aa", "cdr3_aa", "fwr4_aa"])
            
            # Filter based on length requirements
            for column, (min_length, max_length) in length_ranges.items():
                chunk = chunk[chunk[column].apply(lambda x: min_length <= len(str(x)) <= max_length)]
            
            # Append to the output file
            chunk.to_csv(f_out, mode='a', index=False, header=first_chunk)
            first_chunk = False

def get_sequences_per_individual_old(oas_overview, sequences):
    """
    For all sequences in the sequences CSV file, get the file ID
    and associate it with the individual it comes from using the OAS overview.
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
    """
    for individual, files in files_per_individual.items():
        for file in files:
            count = sequences[sequences["File_ID"] == file].shape[0]
            print(f"{individual}: {file} - {count}")
    
    print(files_per_individual)
    """

    return files_per_individual

def get_sequences_per_individual(oas_overview, sequences):
    """
    For all sequences in the sequences CSV file, get the file ID
    and associate it with the individual it comes from using the OAS overview.
    """
    # Read the OAS overview CSV file
    oas_overview = pd.read_csv(oas_overview)
    # Read the sequences CSV file
    sequences = pd.read_csv(sequences)
    
    # Get all unique files from the sequences CSV file
    indices = sequences["File_ID"].unique()
    oas_overview = oas_overview[oas_overview["File_index"].isin(indices)]
    
    files_per_individual = {}
    
    for _, row in oas_overview.iterrows():
        individual = row["Subject"]
        file_index = row["File_index"]
        publication = re.sub(r'\W+', '_', row["Author"]) if "Author" in row else "Unknown"
        
        # If individual is "no", separate entries by publication
        if individual == "no":
            key = f"no_{publication}"
        else:
            key = individual
        
        if key not in files_per_individual:
            files_per_individual[key] = set()
        
        files_per_individual[key].add(file_index)
    
    # Convert sets to lists before returning
    return {key: list(files) for key, files in files_per_individual.items()}

def separate_individuals(sequences, files_per_individual, output_folder):
    """
    Create a file for each individual. 
    Write the sequences from the sequences CSV file that belong to that individual to the new file.
    """
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
    For all sequences in the sequences CSV file, get the file ID
    and associate it with the publication it comes from using the OAS overview.
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
    """
    Create a file for each publication. 
    Write the sequences from the sequences CSV file that belong to that individual to the new file.
    """
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
    """
    Create an overview of the number of sequences in each file (e.g. for sequences separated by individual or publication).
    This can help in setting the number of sequences to sample from each group.
    """
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

def round_robin_sampling(files, total_sequences, output_file):
    """
    Samples sequences as close to evenly as possible from multiple CSV files by 
    sampling equally from files and successively excluding exhausted files.
    Also creates a bar plot showing how many sequences were sampled from each file.

    Args:
    - files: List of CSV files to sample from. Each file could for example represent data from one individual.
    - total_sequences: Total number of sequences to sample.
    - output_file: Path to the output file where sampled sequences are saved.
    """

    file_iters = {file: iter(pd.read_csv(file, chunksize=1)) for file in files}
    
    # Track sequences sampled from each file
    samples_per_file = {basename(file): 0 for file in files}

    with open(output_file, 'w', newline='') as f_out:
        header_written = False

        sampled_count = 0
        active_files = set(files)
    
        while sampled_count < total_sequences and active_files:
            for file in list(active_files):
                try:
                    chunk = next(file_iters[file])

                    # Write to file
                    chunk.to_csv(f_out, header=not header_written, index=False, mode='a')
                    header_written = True

                    # Update count
                    samples_per_file[basename(file)] += 1
                    sampled_count += 1
                
                    if sampled_count >= total_sequences:
                        break
                except StopIteration:
                    active_files.remove(file)  # Remove exhausted file

    # Generate bar plot for sampling distribution
    plot_sampling_distribution(samples_per_file, output_file)


def plot_sampling_distribution(samples_per_file, output_file):
    """
    Generates and saves a bar plot of sampled sequences per file.

    Args:
    - samples_per_file: Dictionary with filenames as keys and number of sampled sequences as values.
    - output_file: Path to file in which the bar plot is saved.
    """
    plt.figure(figsize=(60, 12))

    # Sort for better visualization
    sorted_items = sorted(samples_per_file.items(), key=lambda x: x[1], reverse=True)
    files_sorted, counts_sorted = zip(*sorted_items) if sorted_items else ([], [])

    # Plot bar chart
    bars = plt.bar(range(len(files_sorted)), counts_sorted, color=np.random.rand(len(files_sorted), 3))

    # Annotate values on bars
    for i, bar in enumerate(bars):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1, 
                 f'{bar.get_height()}', ha='center', va='bottom')

    # Labels and title
    plt.xlabel('Source Files')
    plt.ylabel('Number of Sequences Sampled')
    plt.title('Round Robin Sampling Distribution')

    # Format x-axis labels for readability
    plt.xticks(range(len(files_sorted)), files_sorted, rotation=90, ha='right')
    #plt.yticks(fontsize=4)
    plt.tight_layout()

    # Save plot next to output file
    plot_file = os.path.join(dirname(output_file), 'sampling_distribution.png')
    plt.savefig(plot_file)
    plt.close()

def tokenize(training_file, val_file, test_file, cache_dir, out_dir, tokenizer_path):
    """
    Tokenizes the training, validation and test sets and saves them in a given directory.
    Uses the RobertaTokenizer from Hugging Face.
    
    Args:
    - training_file: Path to the training file.
    - val_file: Path to the validation file.
    - test_file: Path to the test file.
    - cache_dir: Directory to cache the datasets.
    - out_dir: Directory to save the tokenized datasets.
    - tokenizer_path: Path to the tokenizer model.
    """
    tokenizer = RobertaTokenizer.from_pretrained(tokenizer_path)
    tokenized_path = os.path.join(out_dir, "tokenized")
    if not os.path.exists(tokenized_path):
        os.makedirs(tokenized_path)

    text_datasets = {
        "train": [training_file],
        "eval": [val_file],
        "test": [test_file]
    }

    # Load dataset
    dataset = load_dataset("text", data_files=text_datasets, cache_dir=cache_dir)

    tokenized_dataset = dataset.map(
        lambda z: tokenizer(
            z["text"],
            padding="max_length",
            truncation=True,
            max_length=150,
            return_special_tokens_mask=True,
        ),
        batched=True,
        num_proc=12,
        remove_columns=["text"],
    )

    # Save to disk
    tokenized_dataset.save_to_disk(out_dir)
    print(f"Tokenized datasets saved to: {out_dir}")

