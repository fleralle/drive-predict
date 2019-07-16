"""Data util librairy."""
import numpy as np
import h5py
import pandas as pd
import os
import fnmatch


def load_dataset_as_dataframe(data_dir_path: str):
    """Load HD5 dataset as a panda dataframe.

    Parameters
    ----------
    data_dir_path : str
        Path to data dir.

    Returns
    -------
    pandas.DataFrame
        Dataset in DataFrame format.

    """
    # Check file exists
    if not os.path.exists(data_dir_path):
        raise FileNotFoundError('Path "{}" not found'.format(data_dir_path))

    dataframe_list = []

    for filename in os.listdir(data_dir_path):
        if fnmatch.fnmatch(filename, '*-*-*--*-*-*.h5'):
            # Load dataset and initialise Dataframe
            with h5py.File(os.path.join(data_dir_path, filename), 'r') as hdf:
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

                dataframe_list.append(df)
    return pd.concat(dataframe_list, ignore_index=True)


def extract_brakes(df: pd.DataFrame, period=100, treshold=-2):
    # brake_df = df.expanding(period).std()
    # brake_df = brake_df.take(range(0, len(df), period)).dropna()
    # brake_indexes = df[df.car_accel < 0].index
    brake_df = df[df.car_accel < treshold]
    # brake_df = df.take(range(0, len(df), period))
    return brake_df


def extract_brake_features(df: pd.DataFrame):
    pass
