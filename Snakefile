configfile: "config.yaml"

from data_functions import select_files
import pandas as pd

rule all:
    input:
        "outputs/sequences.csv"

rule select_files_to_download:
    output:
        "outputs/data_to_download.txt"
    params:
        filters = config["filters"],
    run:
        select_files(filters = params.filters)

rule download_data:
    input:
        data_list = "outputs/data_to_download.txt",
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
        shell("bash download2.sh {input.data_list} {output} 1 {col_numbers}")