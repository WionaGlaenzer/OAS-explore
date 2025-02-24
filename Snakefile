configfile: "config.yaml"

import pandas as pd
from data_functions import (select_files, csv_to_fasta, filter_representative_sequences, 
                          process_anarci_column, get_sequences_per_individual, separate_individuals)
import glob
import os

output_dir = config["output_dir"]
linclust_dir = config["linclust_dir"]
download_dir = config["download_dir"]

rule all:
    input:
        #f"{output_dir}/sequences.csv",
        #f"{output_dir}/data_to_download.csv",
        #f"{output_dir}/sequences.fasta",
        f"{linclust_dir}/antibody_DB_clu_rep.fasta",
        f"{output_dir}/sequences_filtered.csv",
        f"{output_dir}/sequences_filtered_processed.csv",
        f"{output_dir}/sampled_sequences.csv",
        directory(f"{output_dir}/sequences_per_individual/")

rule select_files_to_download:
    output:
        f"{output_dir}/data_to_download.csv"
    params:
        filters = config["filters"],
    run:
        select_files(filters = params.filters, output_file = output[0])

rule download_data:
    input:
        data_list = f"{output_dir}/data_to_download.csv",
        header = "assets/header.csv"
    output:
        f"{output_dir}/sequences.csv"
    params:
        columns_to_keep = config["columns_to_keep"],
        download_dir = config["download_dir"],
        n_lines = config["n_lines"]
    run:
        # Read the header file to get column positions
        header_df = pd.read_csv(input.header)
        # Get the column numbers (adding 1 since positions are 0-based in pandas)
        col_positions = [str(header_df.columns.get_loc(col) + 1) for col in params.columns_to_keep]
        # Join the column numbers with commas
        col_numbers = ",".join(col_positions)
        # Run the shell command
        shell("bash download.sh {input.data_list} {output} {params.n_lines} {col_numbers} {params.download_dir}")

rule csv_to_fasta:
    input:
        sequences_csv = f"{output_dir}/sequences.csv",
    output:
        sequences_fasta = f"{output_dir}/sequences.fasta"
    run:
        csv_to_fasta(input.sequences_csv, output.sequences_fasta)

rule linclust:
    input:
        sequences_fasta = f"{output_dir}/sequences.fasta"
    output:
        sequences_fasta = f"{linclust_dir}/antibody_DB_clu_rep.fasta"
    run:
        shell("bash linclust.sh {linclust_dir} {input.sequences_fasta}")

rule select_filtered_sequences_in_csv:
    input:
        sequences_fasta = f"{linclust_dir}/antibody_DB_clu_rep.fasta",
        sequences_csv = f"{output_dir}/sequences.csv"
    output:
        filename = f"{output_dir}/sequences_filtered.csv"
    run:
        filter_representative_sequences(
            input.sequences_fasta,
            input.sequences_csv,
            output.filename
        )

rule process_anarci_column:
    input:
        sequences_csv = f"{output_dir}/sequences_filtered.csv"
    output:
        filename = f"{output_dir}/sequences_filtered_processed.csv"
    run:
        process_anarci_column(input.sequences_csv, output.filename)

rule sample_sequences:
    input:
        f"{output_dir}/sequences_filtered_processed.csv"
    output:
        f"{output_dir}/sampled_sequences.csv"
    params:
        sampling_scheme = config["sampling_scheme"],
        total_sequences = config["total_sequences"]
    run:
        if params.sampling_scheme == "random":
            shell("bash sample_sequences.sh {input} {output} {params.total_sequences}")
            
        elif params.sampling_scheme == "balance_individuals":
            # Create output directory if it doesn't exist
            shell(f"mkdir -p {output}")
            no_sequences_per_individual = config["total_sequences"]/len(input)
            # Iterate over all input files
            for file in input:
                file_name = os.path.basename(file)
                output_file = f"{output}/{file_name}_sampled"
            
                # Call the shell script to process the file
                shell(f"bash sample_sequences.sh {file} {output_file} {params.no_sequences_per_individual}")

rule get_sequences_per_individual:
    input:
        sequences_csv = f"{output_dir}/sequences_filtered_processed.csv",
        oas_overview = "assets/OAS_overview.csv"
    output:
        directory(f"{output_dir}/sequences_per_individual/")
    run:
        if params.sampling_scheme == "balance_individuals":
            files_per_individual = get_sequences_per_individual(input.oas_overview, input.sequences_csv)
            os.makedirs(output[0], exist_ok=True)
            separate_individuals(input.sequences_csv, files_per_individual, output[0])

rule sample_by_individual:
    input:
        lambda wildcards: glob.glob(f"{output_dir}/sequences_per_individual/*.csv")  # Get all CSV files in directory
    output:
        directory(f"{output_dir}/sampled_sequences_by_individual/")
    params:
        n_samples = config["n_samples"],
        input_dir = f"{output_dir}/sequences_per_individual"  # Add input directory as parameter
    run:
        if params.sampling_scheme == "balance_individuals":
            # Create output directory if it doesn't exist
            shell(f"mkdir -p {output}")
            no_sequences_per_individual = config["total_sequences"]/len(input)
            # Iterate over all input files
            for file in input:
                file_name = os.path.basename(file)
                output_file = f"{output}/{file_name}"
            
                # Call the shell script to process the file
                shell(f"bash sample_sequences.sh {file} {output_file} {params.no_sequences_per_individual}")

