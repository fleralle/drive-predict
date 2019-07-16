"""Data util librairy."""
import numpy as np
import h5py
import pandas as pd
import os


def load_dataset_as_dataframe(file_path: str):
    """Load HD5 dataset as a panda dataframe.

    Parameters
    ----------
    file_path : str
        Path to dataset.

    Returns
    -------
    pandas.DataFrame
        Dataset in DataFrame format.

    """
    # Check file exists
    if not os.path.exists(file_path):
        raise FileExistsError(file_path)

    # Load dataset and initialise Dataframe
    with h5py.File(file_path, 'r') as hdf:
        ls = list(hdf.keys())
        df = pd.DataFrame()
        for key in ls:
            # All but no UN_ variables that are linked to imageries
            if not key.startswith('UN_'):
                key_data = hdf.get(key)
                key_shape = key_data.shape

                # Make sure to assign correctly multi-valued columns
                if len(key_shape) == 1:
                    df[key] = np.array(key_data)
                else:
                    for i in range(key_shape[1]):
                        multi_key_name = key + '_' + str(i)
                        df[multi_key_name] = np.array(key_data[:, i])

        return df


def extract_brakes(df: pd.DataFrame, period=100):
    brake_df = df.expanding(period).mean()
    brake_df = brake_df.take(range(0, len(df), period)).dropna()
    return brake_df
