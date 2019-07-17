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

        if len(brake_df[min_bound:max_bound]) > 10:
            brake_features.append(brake_df[min_bound:max_bound])

    return brake_features


def calculate_distance(origin_coord: np.array, destination_coord: np.array):
    square_distance = sum((destination_coord - origin_coord)**2)
    return (square_distance**.5)*10


def get_coordinates(observation):
    # print(observation.velodyne_gps_0, observation.velodyne_gps_1, observation.velodyne_gps_2, '\n')
    return np.array([
        observation.velodyne_gps_0,
        observation.velodyne_gps_1,
        observation.velodyne_gps_2])
    # return np.array([
    #     observation.fiber_compass_x,
    #     observation.fiber_compass_y,
    #     observation.fiber_compass_z])

def calculate_brake_distance(feature_df):
    # print(len(feature_df))
    origin_obs = feature_df.iloc[0, :]
    destination_obs = feature_df.iloc[-1, :]

    origin_coord = get_coordinates(origin_obs)
    destination_coord = get_coordinates(destination_obs)

    # return destination_coord
    # print(origin_coord, destination_coord)
    return calculate_distance(origin_coord, destination_coord)


def calculate_brake_metrics(features: list):
    # print(type(feature_df.describe()))
    metrics = [pd.concat([feature.speed.describe(), feature.car_accel.describe()], axis=0) for feature in features]
    # brake_distance = [calculate_brake_distance(feature) for feature in features]
    # max_speed = [feature.speed.max() for feature in features]

    return pd.concat(metrics, axis=1).T.reset_index()
    # return metrics.T.reset_index()
    # return pd.concat([pd.Series(brake_distance), pd.Series(max_speed)], axis=1)
