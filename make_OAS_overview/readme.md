# How to make a new OAS overview table

This data pipeline uses the file *OAS_overview.csv* to access parts of the OAS database. If there are updates to the database, it might be necessary to update *OAS_overview.csv*. To do so, follow these steps:

1. Make sure you have a bulk_download.sh file in you directory. You can use the provided one or download an up to date version from OAS.
2. Run make_overview4.sh. This will take around 24 hours, so run it as a slurm job.
3. Use convert_to_csv.py to create the overview csv.
4. To make overview plots, you can use the analyse_OAS_overview.py script.