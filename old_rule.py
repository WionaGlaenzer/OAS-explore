rule download_data:
    """
    Downloads the files listed in data_to_download.csv successively,
    extracts sequences fulfilling length filtering criteria, and writes them to sequences.csv.
    """
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
        # Run the shell command to download and process the data
        shell("bash download.sh {input.data_list} {output} {params.n_lines} {col_numbers} {params.download_dir}")
        #shell("""
        # Check if the output file contains more than just the header
        #line_count=$(cat {output} | wc -l)
        # If there are no data lines (less than or equal to 1 line), raise an error and stop the pipeline
        #if [ "$line_count" -le 1 ]; then
        #    echo "Error: Output file {output} contains only the header and no data. Stopping the pipeline."
        #    exit 1
        #fi
        #""")
