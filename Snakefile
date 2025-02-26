configfile: "config.yaml"

import pandas as pd
from data_functions import (select_files, csv_to_fasta, filter_representative_sequences, 
                          process_anarci_column, get_sequences_per_individual, separate_individuals)
import glob
import os
import math

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
        sequences_csv = f"{output_dir}/sequences_filtered_processed.csv",
        sampled_sequences_by_individual = f"{output_dir}/sampled_sequences_by_individual/.done" if config["sampling_scheme"] == "balance_individuals" else None
    output:
        f"{output_dir}/sampled_sequences.csv"
    params:
        sampling_scheme = config["sampling_scheme"],
        total_sequences = config["total_sequences"]
    run:
        if params.sampling_scheme == "random":
            shell("bash sample_sequences.sh {input.sequences_csv} {output} {params.total_sequences}")
        elif params.sampling_scheme == "balance_individuals":
            # Combine the individual samples into final output
            shell(f"head -n 1 $(ls {output_dir}/sampled_sequences_by_individual/*.csv | head -n 1) > {output}")
            shell(f"for f in {output_dir}/sampled_sequences_by_individual/*.csv; do tail -n +2 $f >> {output}; done")

rule get_sequences_per_individual:
    input:
        sequences_csv = f"{output_dir}/sequences_filtered_processed.csv",
        oas_overview = "assets/OAS_overview.csv"
    output:
        directory = directory(f"{output_dir}/sequences_per_individual/"),
        flag = touch(f"{output_dir}/sequences_per_individual/.done")
    run:
        files_per_individual = get_sequences_per_individual(input.oas_overview, input.sequences_csv)
        os.makedirs(output.directory, exist_ok=True)
        separate_individuals(input.sequences_csv, files_per_individual, output.directory)

rule sample_by_individual:
    input:
        flag = f"{output_dir}/sequences_per_individual/.done",
        files = lambda wildcards: glob.glob(f"{output_dir}/sequences_per_individual/*.csv")
    output:
        directory = directory(f"{output_dir}/sampled_sequences_by_individual/"),
        flag = touch(f"{output_dir}/sampled_sequences_by_individual/.done")
    params:
        total_sequences = config["total_sequences"]
    run:
        shell(f"mkdir -p {output.directory}")
        n_individuals = len(input.files)
        seqs_per_individual = math.ceil(params.total_sequences / n_individuals)
        
        for file in input.files:
            file_name = os.path.basename(file)
            output_file = f"{output.directory}/{file_name}"
            shell(f"bash sample_sequences.sh {file} {output_file} {seqs_per_individual}")

