configfile: "config.yaml"

import pandas as pd
from data_functions import (select_files, csv_to_fasta, filter_representative_sequences, 
                          process_anarci_column, get_sequences_per_individual, separate_individuals,
                          get_sequences_per_publication, separate_publications, number_of_seqs_overview,
                          csv_to_txt, round_robin_sampling)
import glob
import os
import math
import logging

output_dir = config["output_dir"]
linclust_dir = config["linclust_dir"]
download_dir = config["download_dir"]

os.makedirs(output_dir, exist_ok=True)

# Configure logging
logging.basicConfig(
    filename=f"{output_dir}/pipeline.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

rule all:
    input:
        #f"{output_dir}/sequences.csv",
        #f"{output_dir}/data_to_download.csv",
        f"{output_dir}/sequences.fasta",
        f"{linclust_dir}/antibody_DB_clu_rep.fasta",
        f"{output_dir}/sequences_filtered.csv",
        f"{output_dir}/sequences_filtered_processed.csv",
        f"{output_dir}/number_of_seqs_per_individual.csv",
        #f"{output_dir}/sampled_sequences.csv",
        #f"{output_dir}/test_set.csv",
        #f"{output_dir}/training_set.csv",
        #f"{output_dir}/training.txt",
        #"/REDACTED/PATHratch/REDACTED/PATHne_outputs_briney2019/training.txt"
        #f"{output_dir}/download_progress.txt",
        f"{output_dir}/sampled_sequences_round_robin.csv",
        #directory(f"{output_dir}/model/"),
        f"{output_dir}/sequences_per_individual/.done"

rule select_files_to_download:
    """
    Selects which files to download from OAS based on filters from the config file.
    The links to these files are then listed in data_to_download.csv.
    """
    output:
        f"{output_dir}/data_to_download.csv"
    params:
        filters = config["filters"],
    run:
        select_files(filters = params.filters, output_file = output[0])

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

rule csv_to_fasta:
    """
    Converts the sequences.csv file to a FASTA file, which is used as the input for linclust.
    """
    input:
        sequences_csv = f"{output_dir}/sequences.csv",
    output:
        sequences_fasta = f"{output_dir}/sequences.fasta"
    run:
        csv_to_fasta(input.sequences_csv, output.sequences_fasta)

rule linclust:
    """
    Runs linclust on the sequences.fasta file to cluster the sequences 
    and select a representative sequence for each cluster. 
    The coverage and similarity thresholds are set in the config file.
    """
    input:
        sequences_fasta = f"{output_dir}/sequences.fasta"
    output:
        sequences_fasta = f"{linclust_dir}/antibody_DB_clu_rep.fasta"
    params:
        similarity = config["linclust"]["similarity"],
        coverage = config["linclust"]["coverage"]
    run:
        shell("bash linclust.sh {linclust_dir} {input.sequences_fasta} {params.similarity} {params.coverage}")

rule select_filtered_sequences_in_csv:
    """
    Filters the sequences in the sequences.csv file to only include the representative sequences selected by linclust.
    """
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
    """
    Processes the ANARCI column in the sequences_filtered.csv file to turn them into a list, 
    also completes a second round of length filtering.
    """
    input:
        sequences_csv = f"{output_dir}/sequences_filtered.csv"
    output:
        filename = f"{output_dir}/sequences_filtered_processed.csv"
    run:
        process_anarci_column(input.sequences_csv, output.filename)

rule sample_sequences:
    """
    Samples sequences from the sequences_filtered_processed.csv file
    based on the sampling scheme specified in the config file.
    """
    input:
        sequences_csv = f"{output_dir}/sequences_filtered_processed.csv",
        sampled_sequences_by_individual = f"{output_dir}/sampled_sequences_by_individual/.done" if config["sampling_scheme"] == "balance_individuals" else [],
        sampled_sequences_by_publication = f"{output_dir}/sampled_sequences_by_publication/.done" if config["sampling_scheme"] == "balance_publications" else []
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
        elif params.sampling_scheme == "balance_publications":
            shell(f"head -n 1 $(ls {output_dir}/sampled_sequences_by_publication/*.csv | head -n 1) > {output}")
            shell(f"for f in {output_dir}/sampled_sequences_by_publication/*.csv; do tail -n +2 $f >> {output}; done")

rule get_sequences_per_individual:
    """
    Separates the sequences in the sequences_filtered_processed.csv file into separate files for each individual
    based on the OAS overview file to use for sampling balanced by individuals.
    """
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
    """
    Samples the same number of sequences from each individual to reach a total number of sequences specified in the config file.
    """
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

rule get_sequences_per_publication:
    """
    Separates the sequences in the sequences_filtered_processed.csv file into separate files for each publication
    based on the OAS overview file to use for sampling balanced by publications.
    """
    input:
        sequences_csv = f"{output_dir}/sequences_filtered_processed.csv",
        oas_overview = "assets/OAS_overview.csv"
    output:
        directory = directory(f"{output_dir}/sequences_per_publication/"),
        flag = touch(f"{output_dir}/sequences_per_publication/.done")
    params:
        total_sequences = config["total_sequences"]
    run:
        seqs_per_publication = get_sequences_per_publication(input.oas_overview, input.sequences_csv)
        files_per_publication = seqs_per_publication[0]
        sample_per_publication = math.ceil(params.total_sequences / len(files_per_publication))
        enough_sequences = True
        print(seqs_per_publication)
        for publication, n in seqs_per_publication[1].items():
            if n < sample_per_publication:
                logging.warning(f"Publication {publication} has only {n} sequences, less than the desired {sample_per_publication}")
                enough_sequences = False
        if not enough_sequences:
            logging.error("Not enough sequences to sample by publication")
            raise ValueError("Not enough sequences to sample by publication")
        os.makedirs(output.directory, exist_ok=True)
        separate_publications(input.sequences_csv, files_per_publication, output.directory)

rule sample_by_publication:
    """
    Samples the same number of sequences from each publication to reach a total number of sequences specified in the config file.
    """
    input:
        flag = f"{output_dir}/sequences_per_publication/.done",
        files = lambda wildcards: glob.glob(f"{output_dir}/sequences_per_publication/*.csv")
    output:
        directory = directory(f"{output_dir}/sampled_sequences_by_publication/"),
        flag = touch(f"{output_dir}/sampled_sequences_by_publication/.done")
    params:
        total_sequences = config["total_sequences"]
    run:
        shell(f"mkdir -p {output.directory}")
        n_publications = len(input.files)
        seqs_per_publication = math.ceil(params.total_sequences / n_publications)
        
        for file in input.files:
            file_name = os.path.basename(file)
            output_file = f"{output.directory}/{file_name}"
            shell(f"bash sample_sequences.sh {file} {output_file} {seqs_per_publication}")

rule number_of_seqs_overview:
    """
    Creates an overview of the number of sequences per individual based on the sampled sequences.
    This can be used to choose a reasonable number of sequences to sample.
    """
    input:
        flag = f"{output_dir}/sequences_per_individual/.done",
        files = lambda wildcards: glob.glob(f"{output_dir}/sequences_per_individual/*.csv")
    output:
        f"{output_dir}/number_of_seqs_per_individual.csv"
    run:
        number_of_seqs_overview(input.files, output[0])

rule split_data:
    """
    Splits the data into training, validation and test sets.
    """
    input:
        sequences_csv = f"{output_dir}/sampled_sequences.csv"
    output:
        training = f"{output_dir}/training_set.csv",
        validation = f"{output_dir}/validation_set.csv",
        test = f"{output_dir}/test_set.csv"
    params:
        training_fraction = config["training_fraction"],
        validation_fraction = config["validation_fraction"]
    run:
        shell(f"bash split_data.sh {input.sequences_csv} {output.training} {output.validation} {output.test} {params.training_fraction} {params.validation_fraction} 0 {output_dir}")

rule csv_to_txt:
    """
    Converts the training, validation, and test sets to txt files containing only the sequences without metadata.
    """
    input:
        training = f"{output_dir}/training_set.csv",
        validation = f"{output_dir}/validation_set.csv",
        test = f"{output_dir}/test_set.csv"
    output:
        training = f"{output_dir}/training.txt",
        validation = f"{output_dir}/validation.txt",
        test = f"{output_dir}/test.txt"
    run:
        csv_to_txt(input.training, output.training)
        csv_to_txt(input.validation, output.validation)
        csv_to_txt(input.test, output.test)

rule model_training:
    input:
        training = f"{output_dir}/training.txt",
        validation = f"{output_dir}/validation.txt",
        test = f"{output_dir}/test.txt"
    params:
        cache_dir = config["cache_dir"],
        model_name = config["model_name"],
        environment = config["training_environment"],
        tokenizer = config["tokenizer"],
        deepspeed_config = "assets/deepspeed_config.json",
        wandb_base_dir = config["wandb_base_dir"]
    output:
        directory = directory(f"{output_dir}/model/"),
        flag = f"{output_dir}/model/.done"  # This flag file marks completion
    shell:
        '''
        source {params.environment}/bin/activate
        
        mkdir -p {output.directory}

        RPORT=$((RANDOM % 55535 + 10000))
        
        bash submit_and_wait.sh "
        echo 'CUDA_HOME is set to: $CUDA_HOME'
        echo \\"Using rendezvous port: $RPORT\\"
        export WANDB_DIR='{params.wandb_base_dir}';
        torchrun --nproc_per_node=auto --rdzv-backend=c10d --rdzv-endpoint=\\"localhost:$RPORT\\" \
        train_model.py {input.training} {input.validation} {input.test} {params.cache_dir} \
        {output.directory} {params.model_name} {output.flag} {params.tokenizer} {params.deepspeed_config}"
        '''

rule sample_by_publication_round_robin:
    """
    Samples sequences in a round-robin fashion from each publication until the total number of sequences specified in the config file is reached.
    """
    input:
        flag = f"{output_dir}/sequences_per_individual/.done",
        files = lambda wildcards: glob.glob(f"{output_dir}/sequences_per_individual/*.csv")
    output:
        file = f"{output_dir}/sampled_sequences_round_robin.csv",
        flag = touch(f"{output_dir}/sampled_sequences_by_individual_round_robin/.done")
    params:
        total_sequences = config["total_sequences"]
    run:
        shell(f"mkdir -p $(dirname {output.file})")
        round_robin_sampling(input.files, params.total_sequences, output.file)

rule tokenize():
    """
    Tokenizes the sequences in the training, validation, and test sets using the specified tokenizer.
    """
    input:
        training_txt = f"{output_dir}/training.txt",
        validation_txt = f"{output_dir}/validation.txt",
        test_txt = f"{output_dir}/test.txt"
    params:
        tokenizer = config["tokenizer"],
        cache_dir = config["cache_dir"],
        tokenized_folder = f"{output_dir}/tokenized"
    output:
        tokenized_dict = directory(f"{output_dir}/tokenized/dataset_dict.json"),
    run:
        shell(f"mkdir -p {output.tokenized_folder}")
        shell(f"python pre_tokenize.py {input.training_txt} {input.validation_txt} {input.test_txt} {params.cache_dir} {params.tokenized_folder} {params.tokenizer}")

rule model_training_after_tokenization():
    """
    Trains the model using pre-tokenized sequences.
    TODO: create a submit_training_job.sh script with parameters.
    """
    input:
        tokenized_dict = f"{output_dir}/tokenized/dataset_dict.json"
        tokenized_folder = f"{output_dir}/tokenized"
    params:
        cache_dir = config["cache_dir"],
        model_name = config["model_name"],
        environment = config["training_environment"],
        deepspeed_config = config["deepspeed_config"],
        wandb_base_dir = config["wandb_base_dir"]
    output:
        directory = directory(f"{output_dir}/model/"),
        flag = f"{output_dir}/model/.done"  # This flag file marks completion
    shell:
        bash submit_training_job_with_parameters.sh