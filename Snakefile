configfile: "config.yaml"

from data_functions import select_files

rule select_files_to_download:
    output:
        "data_to_download.txt"
    params:
        species = config["species"],
        publications = config["publications"]
    run:
        select_files(species = params.species, publications = params.publications)

rule download_data:
    input:
        "data_to_download.txt"
    output:
        touch("download_complete.txt")