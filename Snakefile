configfile: "config.yaml"

from data_functions import select_files

rule select_files_to_download:
    output:
        "outputs/data_to_download.txt"
    params:
        filters = config["filters"],
    run:
        select_files(filters = params.filters)

rule download_data:
    input:
        "data_to_download.txt"
    output:
        touch("download_complete.txt")