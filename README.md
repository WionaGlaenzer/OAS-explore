# plm_training_pipeline
A snakemake pipeline for training antibody language models

## Rules of the pipeline

The rulegraph shows the rules:

<img src="rulegraph.png" width="300" alt="Rule graph">

## Creating an environment to run the pipeline:

On Euler:
module load stack/2024-06 gcc/12.2.0
module load python/3.11.6 cuda/12.1.1 ninja/1.11.1
python3.11 -m venv pipeline
source pipeline/bin/activate
pip install -r requirements.txt