# Antibody PLM Training Pipeline
A snakemake pipeline for training antibody language models on data from the OAS.

## Rules of the pipeline

The rulegraph shows the rules:

<img src="rulegraph.png" width="300" alt="Rule graph">

## Setting up the pipeline

On Euler:
module load stack/2024-06 gcc/12.2.0
module load python/3.11.6 cuda/12.1.1 ninja/1.11.1
python3.11 -m venv pipeline
source pipeline/bin/activate
pip install -r requirements.txt

Linclust has to be installed separately following these instructions: https://github.com/soedinglab/mmseqs2/wiki#install-mmseqs2-for-linux
On Euler we followed the steps listed in the section "Compile from source under Linux".

Model training requires an additional environment with the necessary packages.

## Run the pipeline

Use the command *snakemake* to run the pipeline.

Required computational resources depend on the filtering choices. 
Downloading and processing most publications from OAS takes approximately 3 days on 1 CPU with 300gb memory.

## Log in to weights and biases for tracking during model training

- Make sure you have created a *training environment* for model training
- Activate the *training environment*
- Run "wandb login"
- Enter the API key from the weights and biases website. If you don't have an account yet, create a free account first.
