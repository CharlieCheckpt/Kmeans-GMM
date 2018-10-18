"""Utilities.
"""
import pandas as pd
import os


DATA_PATH = "./data"

def load_data(data_file="EMGaussian.data", header=None, delimiter=" "):
    """Loads data file..
        data_file (str, optional): Defaults to "EMGaussian.data". Name of data file, must be in "./data".
    
    Returns:
        np.array: loaded data.
    """

    data = pd.read_csv(os.path.join(DATA_PATH, data_file), header=header, delimiter=delimiter)
    data = data.values
    print(f"data {data.shape} loaded")
    return data