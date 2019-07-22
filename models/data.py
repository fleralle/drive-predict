"""Data util librairy."""
import pandas as pd
import os
import fnmatch
import matplotlib.pyplot as plt
import seaborn as sns


# Columns names in CSV file.
TRIP_DATA_COLUMNS = [
    'time',
    'speed',
    'shift',
    'engine_Load',
    'car_accel',
    'rpm',
    'pitch',
    'lateral_acceleration',
    'passenger_count',
    'car_load',
    'ac_status',
    'window_opening',
    'radio_volume',
    'rain_intensity',
    'visibility',
    'driver_wellbeing',
    'driver_rush'
]


def load_dataset_as_dataframe(data_dir_path: str):
    """Load dataset as a panda dataframe.

    Parameters
    ----------
    data_dir_path : str
        Path to data dir.

    Returns
    -------
    pandas.DataFrame
        Dataset in DataFrame format.

    """
    # Initialize returned output
    dataframe_list = []
    max_time = 0

    # Check file exists
    if not os.path.exists(data_dir_path):
        raise FileNotFoundError('Path "{}" not found'.format(data_dir_path))

    for filename in os.listdir(data_dir_path):
        if fnmatch.fnmatch(filename, 'fileID*_ProcessedTripData.csv'):
            # Load dataset and initialise Dataframe
            df = pd.read_csv(os.path.join(data_dir_path, filename), header=None)
            df.columns = TRIP_DATA_COLUMNS

            # Assign time to keep record continuity
            df.time = df.time + max_time
            max_time = df.time.max()
            dataframe_list.append(df)

    # Merge all together
    concatenated_df = pd.concat(dataframe_list, ignore_index=True)

    # Assign idx based on new index. Useful to brakedown events later on.
    concatenated_df['idx'] = concatenated_df.index

    return concatenated_df


def plot_feature_distributions(
                                df: pd.DataFrame,
                                figsize: tuple,
                                features=TRIP_DATA_COLUMNS):
    """Plot features distributions.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing the features.
    figsize : tuple
        Matplotlib figsize param.
    features : list
        List of features.

    """
    nrows = len(features)//2 + 1 if len(features) % 2 else len(features)//2

    fig, axes = plt.subplots(nrows=nrows, ncols=2, figsize=figsize)
    fig.delaxes(axes[nrows-1][1])  # Removes last empty subplot

    for i, col_name in enumerate(features):
        row = i // 2
        col = i % 2
        sns.distplot(df[col_name], ax=axes[row][col])


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
    boundaries = (brake_df.time - brake_df.time.shift()) > (brake_interval)
    new_boundaries = boundaries.reset_index()

    # boundaries_indexes = new_boundaries[new_boundaries['idx']].index
    boundaries_indexes = new_boundaries[new_boundaries['time']].index

    for i in range(len(boundaries_indexes)):
        min_bound = 0 if i == 0 else boundaries_indexes[i-1]
        max_bound = boundaries_indexes[i]

        if len(brake_df[min_bound:max_bound]) > 10:
            brake_features.append(brake_df[min_bound:max_bound])

    return brake_features


def calculate_event_metrics(event_df: pd.DataFrame):
    """Calculate metrics for given driving event.

    Parameters
    ----------
    event_df : pd.DataFrame
        Driving event dataframe containing all car measures.

    Returns
    -------
    pd.Series
        Series containing event metrics.

    """
    # Build numerical data metrics
    numerical_features = [
        'speed',
        'car_accel',
        'lateral_acceleration',
        'rpm',
        'pitch',
        'shift'
    ]
    num_metrics = [event_df[feature].describe().add_prefix(feature + '_')
                   for feature in numerical_features]
    num_metrics_ds = pd.concat(num_metrics, axis=0)

    # Build categorical data metrics
    categorical_features = [
        'driver_rush',
        'visibility',
        'rain_intensity',
        'driver_wellbeing'
    ]
    cat_metrics = [event_df[feature].mean()
                   for feature in categorical_features]
    cat_metrics_ds = pd.Series(cat_metrics, index=categorical_features)

    # Merge numerical and categorical metrics
    metrics_ds = pd.concat([num_metrics_ds, cat_metrics_ds], axis=0)

    # Clean duplicated 'count' columns and rename labels
    duplicated_cols = [col + '_count' for col in numerical_features[1:]]
    metrics_ds.drop(labels=duplicated_cols, inplace=True)
    metrics_ds.rename({'speed_count': 'observations'}, inplace=True)
    metrics_ds.rename(lambda x: x.replace('%', ''), inplace=True)

    return metrics_ds


def get_events_metrics(events: list):
    """Return events metrics dataframe.

    Parameters
    ----------
    events : list
        List of driving events.

    Returns
    -------
    pd.DataFrame
        DataFrame containing all event metrics.

    """
    metrics = [calculate_event_metrics(event) for event in events]

    # Format dataframe
    metrics_df = pd.concat(metrics, axis=1).T.reset_index()

    metrics_df.drop(columns=['index'], inplace=True)

    return metrics_df
