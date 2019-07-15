"""Data util librairy."""
import numpy as np
import h5py
import pandas as pd
import os


def get_dataset_definition(file_path: str):
    with h5py.File(file_path, 'r') as hdf:
        ls = list(hdf.keys())
        return ls
    return None


def load_dataset_as_dataframe(file_path: str):
    # Check file exists
    if not os.path.exists(file_path):
        raise FileExistsError(file_path)

    with h5py.File(file_path, 'r') as hdf:
        ls = list(hdf.keys())
        df = pd.DataFrame()
        for key in ls:
            print(key)
            # if not key.startswith('UN_') and key not in ['fiber_compass', 'fiber_accel', 'fiber_gyro', 'velodyne_gps', 'velodyne_heading', 'velodyne_imu']:
            if not key.startswith('UN_'):
                key_data = hdf.get(key)
                key_shape = key_data.shape
                print(key_shape)
                print(len(key_shape))
                if len(key_shape) == 1:
                    df[key] = np.array(key_data)
                else:
                    for i in range(key_shape[1]):
                        multi_key_name = key + '_' + str(i)
                        df[multi_key_name] = np.array(key_data[:, i])

        # df.columns = ls
        return df
    # return 'dsd'

def convert_ecef_to_lat_lon_elevation():
    pass

def scale_features():
    pass
