import pandas as pd
from sklearn.model_selection import KFold
import dask.bag as db

def get_CV_splits(data_file, n_splits=3):
    """
    Returns a list of test set definitions (e.g., file paths or indices).
    """
    bag = db.read_text(data_file)
    lines = bag.to_dataframe(columns=['sequence'])
    kf = KFold(n_splits=n_splits, shuffle=False, random_state=42)
    index_values = lines.index.compute()
    splits = []
    for i, test_index in enumerate(kf.split(index_values)):
        test_index_series = db.from_pandas(pd.Series(test_index), npartitions=1)
        test_mask = lines.index.isin(test_index_series)
        test_data_df = lines[test_mask]
        output_file = f"fold_{i}.txt"
        test_data_df.to_csv(output_file, single_file=True) #save each fold to a separate file
        splits.append(output_file) #return the file path
    return splits
