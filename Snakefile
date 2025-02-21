configfile: "config.yaml"

from data_functions import select_files, csv_to_fasta, filter_representative_sequences
import pandas as pd

rule all:
    input:
        # "outputs/sequences.csv",
        "outputs/data_to_download.csv",
        "outputs/sequences.fasta",
        "linclust/antibody_DB_clu_rep.fasta",
        "outputs/sequences_filtered.csv"

rule select_files_to_download:
    output:
        "outputs/data_to_download.csv"
    params:
        filters = config["filters"],
    run:
        select_files(filters = params.filters)

rule download_data:
    input:
        data_list = "outputs/data_to_download.csv",
        header = "assets/header.csv"
    output:
        "outputs/sequences.csv"
    params:
        columns_to_keep = config["columns_to_keep"]
    run:
        # Read the header file to get column positions
        header_df = pd.read_csv(input.header)
        # Get the column numbers (adding 1 since positions are 0-based in pandas)
        col_positions = [str(header_df.columns.get_loc(col) + 1) for col in params.columns_to_keep]
        # Join the column numbers with commas
        col_numbers = ",".join(col_positions)
        # Run the shell command
        shell("bash download.sh {input.data_list} {output} 1 {col_numbers}")

rule csv_to_fasta:
    input:
        sequences_csv = "outputs/sequences.csv",
    output:
        sequences_fasta = "outputs/sequences.fasta"
    run:
        csv_to_fasta(input.sequences_csv, output.sequences_fasta)

rule linclust:
    input:
        sequences_fasta = "outputs/sequences.fasta"
    output:
        sequences_fasta = "linclust/antibody_DB_clu_rep.fasta"
    run:
        shell("bash linclust.sh")

rule select_filtered_sequences_in_csv:
    input:
        sequences_csv = "outputs/sequences.csv",
        sequences_fasta = "linclust/antibody_DB_clu_rep.fasta"
    output:
        "outputs/sequences_filtered.csv"
    run:
        filter_representative_sequences(
        "linclust/antibody_DB_clu_rep.fasta",
        "outputs/sequences.csv",
        "outputs/sequences_filtered.csv"
    )

"""
rule sample_sequences:
    input:
        "outputs/sequences.csv"
    output:
        "outputs/sampled_sequences.csv"
    run:
        shell("bash sample_sequences.sh {input} {output}")
"""