import pandas as pd

def select_files(species, publications):
    df = pd.read_csv('OAS_overview.csv')
    df_selected_species = df[df['Species'].isin(species["include"])]
    df_selected_publication = df_selected_species[df_selected_species['Author'].isin(publications["include"])]

    # Write the Download links of the selected rows into a text file
    with open('data_to_download.txt', 'w') as f:
        for index, row in df_selected_publication.iterrows():
            f.write(row['Download_Link'] + '\n')