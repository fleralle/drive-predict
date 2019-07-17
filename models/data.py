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
    concatenated_df = pd.concat(dataframe_list, ignore_index=True)

    # Reassign idx based on new index
    concatenated_df['idx'] = concatenated_df.index
    return concatenated_df


def extract_brake_events(df: pd.DataFrame, treshold=-2):
    """Extract brake events out of dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing driving measures.
    treshold : int
        Treshold for significant acceleration in m/s^2.

    Returns
    -------
    pd.DataFrame
        Dataframe containing only brake events driving measures.

    """
    brake_df = df[df.car_accel < treshold]
    return brake_df


def extract_brake_features(df: pd.DataFrame, treshold=-2, brake_interval=2):
    """Extract a list of brake event time series.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing driving measures.
    treshold : int
        Treshold for significant acceleration in m/s^2.
    brake_interval : int
        Time interval in seconds where 2 brake events are considered distincts.

    Returns
    -------
    list
        List of DataFrame containing brake events time series.

    """
    # prepare output
    brake_features = []
    brake_df = extract_brake_events(df)

    # Measures are taken at a 100Hz frequency.
    boundaries = (brake_df.idx - brake_df.idx.shift()) > (brake_interval * 100)
    new_boundaries = boundaries.reset_index()

    boundaries_indexes = new_boundaries[new_boundaries['idx']].index

    for i in range(len(boundaries_indexes)):
        min_bound = 0 if i == 0 else boundaries_indexes[i-1]
        max_bound = boundaries_indexes[i]
        brake_features.append(brake_df[min_bound:max_bound])

    return brake_features


def calculate_feature_metrics(feature_df: pd.DataFrame):
    pass
