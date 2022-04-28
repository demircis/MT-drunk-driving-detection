import pandas as pd
import numpy as np
import datetime
from joblib import Parallel, delayed
from tqdm import tqdm

from get_features import get_features

GROUPING_COLUMNS = ['subject_id', 'subject_state', 'subject_scenario']

DRIVER_BEHAVIOR = ['steer', 'gas', 'brake', 'SteerSpeed', 'indicator_left', 'indicator_right',
                    'gas_vel', 'brake_vel', 'gas_acc', 'brake_acc', 'SteerSpeed_acc',
                    'gas_jerk', 'brake_jerk', 'SteerSpeed_jerk', 'SteerError']
VEHICLE_BEHAVIOR = ['velocity', 'acc', 'acc_jerk', 'latvel', 'YawRate', 'latvel_acc', 'latvel_jerk',
                    'YawRate_acc', 'YawRate_jerk']
RADAR = ['lane_position', 'lane_distance_left_edge', 'lane_distance_right_edge', 'lane_crossing', 
         'is_crossing_lane', 'is_crossing_lane_left', 'is_crossing_lane_right',
         'lane_crossing_left', 'lane_crossing_right', 'lane_switching', 'opp_lane_switching',
         'Ttc', 'TtcOpp', 'Thw', 'Dhw']
NAVI = ['dtoint', 'SpeedDif', 'SpeedDif_acc', 'speed_limit_exceeded']

SIGNALS = {'driver_behavior': DRIVER_BEHAVIOR, 'vehicle_behavior': VEHICLE_BEHAVIOR, 'radar': RADAR, 'navi': NAVI}

EVENTS = ['brake', 'brake_to_gas', 'gas', 'gas_to_brake', 'overtaking', 'road_sign', 'turning']

STATS = ['mean', 'std', 'min', 'max', 'q5', 'q95', 'range', 'iqrange', 'iqrange_5_95', 'sum', 'energy',
        'skewness', 'kurtosis', 'peaks', 'rms', 'lineintegral', 'n_above_mean', 'n_below_mean', 'n_sign_changes', 'ptp']

# REMOVE_PER_EVENT = {'brake': ['gas', 'gas_acc', 'gas_jerk', 'gas_vel'],
#                     'brake_to_gas': ['gas', 'gas_acc', 'gas_jerk', 'gas_vel', 'brake'],
#                     'gas': ['brake', 'brake_acc', 'brake_jerk', 'brake_vel'],
#                     'gas_to_brake': ['brake', 'brake_acc', 'brake_jerk', 'brake_vel', 'gas'],
#                     'overtaking': [],
#                     'road_sign': [],
#                     'turning': []
#                     }

SELECTED_SIGNALS = [
    'brake',
    'brake_acc',
    'brake_jerk',
    'brake_vel',
    'gas',
    'gas_acc',
    'gas_jerk',
    'gas_vel',
    'steer',
    'SteerSpeed',
    'SteerSpeed_acc',
    'SteerSpeed_jerk',
    'speed_limit_exceeded',
    'SpeedDif',
    'Dhw',
    'is_crossing_lane_left',
    'is_crossing_lane_right',
    'lane_crossing',
    'lane_distance_left_edge',
    'lane_distance_right_edge',
    'Ttc',
    'TtcOpp',
    'acc',
    'acc_jerk',
    'velocity',
    'latvel_acc',
    'latvel_jerk',
    'YawRate_acc',
    'YawRate_jerk',
    'YawRate'
]

SELECTED_STATS = ['mean', 'std', 'min', 'max', 'q5', 'q95', 'iqrange', 'iqrange_5_95', 'skewness', 'kurtosis', 'peaks', 'rms']

SUM_COLUMNS = ['lane_crossing', 'lane_crossing_left', 'lane_crossing_right', 'is_crossing_lane', 'is_crossing_lane_left', 'is_crossing_lane_right', 'speed_limit_exceeded']


def calc_can_data_features(window_sizes):

    can_data = pd.read_parquet('out/can_data.parquet')

    for key, signal in SIGNALS.items():
        for window_size in window_sizes:
            can_data_features = can_data.groupby(GROUPING_COLUMNS).apply(
                    lambda x: get_features(x[['timestamp'] + signal], window_size * 1000, num_cores=16)
                )
            
            can_data_features.to_parquet('out/can_data_features_{}_windowsize_{}s.parquet'.format(key, window_size))


def filter_can_data_event_columns():
    for event in EVENTS:
        can_data_event = pd.read_parquet('out/can_data_{}_events.parquet'.format(event))
        columns_per_signal = [[signal + '_' + stat for signal in DRIVER_BEHAVIOR + VEHICLE_BEHAVIOR + NAVI + RADAR] for stat in STATS]
        selected_columns = []
        for sublist in columns_per_signal:
            selected_columns += sublist
        can_data_event_filtered = None
        if event == 'road_sign':
            can_data_event_filtered = can_data_event[['duration', 'sign_type'] + selected_columns]
        else:
            can_data_event_filtered = can_data_event[['duration'] + selected_columns]
        can_data_event_filtered.to_parquet('out/can_data_{}_events_features.parquet'.format(event))


def calc_event_features_in_window(window_sizes):
    for window_size in window_sizes:
        def get_event_info_for_windows(data):
            subject_id = np.unique(data.index.get_level_values('subject_id'))[0]
            subject_state = np.unique(data.index.get_level_values('subject_state'))[0]
            subject_scenario = np.unique(data.index.get_level_values('subject_scenario'))[0]
            window_timestamps = data.index.get_level_values('datetime')
            event_info_for_windows = []
            for event in EVENTS:
                if event == 'turning' and subject_scenario == 'highway':
                    continue
                event_data = pd.read_parquet('out/can_data_{}_events_features.parquet'.format(event))
                event_data = event_data.loc[subject_id, subject_state, subject_scenario, :]
                event_info = []
                def get_event_info_per_window(timestamp, event_info):
                    min_timestamp = timestamp
                    max_timestamp = timestamp + datetime.timedelta(seconds=window_size)
                    start_timestamps = event_data.index.get_level_values('datetime')
                    end_timestamps = event_data.index.get_level_values('datetime') + event_data['duration'].apply(lambda duration: datetime.timedelta(seconds=duration))
                    events_in_window = event_data.loc[(
                        ((start_timestamps >= min_timestamp) & (start_timestamps < max_timestamp)) |
                        ((end_timestamps >= min_timestamp) & (end_timestamps < max_timestamp))
                    )]
                    mean_duration = 0
                    std_duration = 0
                    if not events_in_window['duration'].empty:
                        mean_duration = events_in_window['duration'].mean()
                        std_duration = events_in_window['duration'].std(ddof=0)
                    start_timestamps = events_in_window.index.get_level_values('datetime')
                    end_timestamps = events_in_window.index.get_level_values('datetime') + events_in_window['duration'].apply(lambda duration: datetime.timedelta(seconds=duration))
                    # event_durations_in_window = [np.min(
                    #         DateTimeRange(start_timestamp, max_timestamp),
                    #         DateTimeRange(start_timestamp, start_timestamp + datetime.timedelta(seconds=events_in_window['duration']))
                    #     ) for start_timestamp in start_timestamps]
                    # overlap_duration = 0
                    # for i in range(len(event_durations_in_window)-1):
                    #     event_duration_i = event_durations_in_window[i]
                    #     for j in range(i+1, len(event_durations_in_window)):
                    #         event_duration_j = event_durations_in_window[j]
                    #         if event_duration_i.is_intersection(event_duration_j):
                    #             overlap_duration += event_duration_i.intersection(event_duration_j).timedelta.total_seconds()
                    # total_ratio = (np.sum(event_durations_in_window) - overlap_duration) / window_size
                    event_durations = events_in_window['duration'].to_numpy()
                    mean_ratio = 0
                    std_ratio = 0
                    if not start_timestamps.empty:
                        ratios = np.minimum(
                            np.minimum(
                                (max_timestamp - start_timestamps).total_seconds().to_numpy(), (end_timestamps - min_timestamp).apply(lambda x: x.total_seconds()).to_numpy()
                            ), event_durations) / window_size
                        mean_ratio = np.mean(ratios)
                        std_ratio = np.std(ratios)
                    count = len(events_in_window.index)
                    event_info_dict = {
                        'mean_duration': mean_duration,
                        'std_duration': std_duration,
                        'mean_ratio': mean_ratio,
                        'std_ratio': std_ratio,
                        'count': count
                    }
                    events_in_window_timestamps = events_in_window.index.get_level_values('datetime')
                    feature_window_size = (events_in_window_timestamps.max() - events_in_window_timestamps.min()).total_seconds() * 1000
                    if not np.isnan(feature_window_size):
                        feature_window_size = round(feature_window_size)
                    events_in_window_subset = events_in_window[select_columns(events_in_window)].copy()
                    features_dict = None
                    if events_in_window_subset.empty:
                        features_dict = events_in_window_subset.to_dict('list')
                        features_dict = {event + '_' + key: [0] for key, _ in features_dict.items()}
                    else:
                        events_in_window_subset.loc[:, 'timestamp'] = events_in_window_timestamps
                        features = get_features(events_in_window_subset, feature_window_size, num_cores=1, step_size=str(feature_window_size + 1) + 'ms', disable_progress_bar=True)
                        features_dict = features.to_dict('list')
                    features_dict = {event + '_' + key: values[0] for key, values in features_dict.items()}
                    event_info_dict = {event + '_' + key: value for key, value in event_info_dict.items()}
                    event_info_dict.update(features_dict)
                    event_info.append(event_info_dict)
                    return event_info
                event_info = Parallel(n_jobs=8)(delayed(get_event_info_per_window)(timestamp, event_info) for timestamp in tqdm(window_timestamps))
                event_info_for_windows.append(pd.DataFrame(event_info).set_index(window_timestamps))
            result = pd.concat(event_info_for_windows, axis=1)
            return result

        can_data_features = pd.read_parquet('out/can_data_features_vehicle_behavior_windowsize_{}s.parquet'.format(window_size))
        events_per_window = can_data_features.groupby(['subject_id', 'subject_state', 'subject_scenario']).apply(get_event_info_for_windows)
        events_per_window.to_parquet('out/can_data_events_per_window_windowsize_{}s.parquet'.format(window_size))
        print('done')


def select_columns(data):
    stat_columns_list = [
        [col for col in data.columns if col in 
            [sig + '_' + s for s in (SELECTED_STATS + ['sum'] if sig in SUM_COLUMNS else SELECTED_STATS)]
        ] for sig in SELECTED_SIGNALS
    ]
    stat_columns = []
    for item in stat_columns_list:
        stat_columns += item
    return stat_columns