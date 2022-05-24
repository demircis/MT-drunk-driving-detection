import pandas as pd
import numpy as np
import warnings
from event_functions import calculate_event_stats
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
NAVI = ['dtoint', 'SpeedDif', 'speed_limit_exceeded', 'turn_angle']

SIGNALS = {'driver_behavior': DRIVER_BEHAVIOR, 'vehicle_behavior': VEHICLE_BEHAVIOR, 'radar': RADAR, 'navi': NAVI}

EVENTS = ['brake', 'brake_to_gas', 'gas', 'gas_to_brake', 'overtaking', 'road_sign', 'turning']

STATS = ['mean', 'std', 'min', 'max', 'q5', 'q95', 'range', 'iqrange', 'iqrange_5_95', 'sum', 'energy',
        'skewness', 'kurtosis', 'peaks', 'rms', 'lineintegral', 'n_above_mean', 'n_below_mean', 'n_sign_changes', 'ptp']

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
    'is_crossing_lane',
    'lane_crossing',
    'lane_distance_left_edge',
    'lane_distance_right_edge',
    'lane_position',
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
    print('Calculate CAN data features')
    can_data = pd.read_parquet('out/can_data.parquet')

    for key, signal in SIGNALS.items():
        for window_size in window_sizes:
            can_data_features = can_data.groupby(GROUPING_COLUMNS).apply(
                    lambda x: get_features(x[['timestamp'] + signal], window_size * 1000, num_cores=30)
                )
            
            can_data_features.to_parquet('out/can_data_features_{}_windowsize_{}s.parquet'.format(key, window_size))
    print('done')


def calc_can_data_event_features():
    print('Calculate CAN data event features')
    can_data = pd.read_parquet('out/can_data.parquet')
    for event in EVENTS:
        print('event: {}'.format(event))
        columns_per_signal = [[signal + '_' + stat for signal in DRIVER_BEHAVIOR + VEHICLE_BEHAVIOR + NAVI + RADAR] for stat in STATS]
        selected_columns = []
        for sublist in columns_per_signal:
            selected_columns += sublist
        can_data_event_features = can_data.groupby(GROUPING_COLUMNS).apply(lambda x: calculate_event_stats(x, event))
        can_data_event_features.index.rename('datetime', level=3, inplace=True)
        if event == 'road_sign':
            can_data_event_features = can_data_event_features[['duration', 'sign_type'] + selected_columns]
        else:
            can_data_event_features = can_data_event_features[['duration'] + selected_columns]
        can_data_event_features.to_parquet('out/can_data_{}_events_features.parquet'.format(event))
    print('done')


def calc_event_features_in_window(window_sizes):
    print('Calculate event features in sliding windows')
    for window_size in window_sizes:
        window_timestamps = pd.read_parquet('out/can_data_features_vehicle_behavior_windowsize_{}s.parquet'.format(window_size)).reset_index(level='datetime')['datetime']
        for event in EVENTS:
            print('event: {}'.format(event))
            columns_per_signal = [[signal + '_' + stat for signal in SELECTED_SIGNALS] for stat in STATS]
            selected_columns = []
            for sublist in columns_per_signal:
                selected_columns += sublist
            can_data_event_features = pd.read_parquet('out/can_data_{}_events_features.parquet'.format(event), columns=['duration'] + selected_columns)
            events_per_window = can_data_event_features.groupby(GROUPING_COLUMNS).apply(lambda data: get_event_info_for_windows(data, window_size, window_timestamps, event))
            events_per_window.to_parquet('out/can_data_{}_events_per_window_windowsize_{}s.parquet'.format(event, window_size))
    print('done')


def get_event_info_for_windows(data, window_size, window_timestamps, event):
    subject_id = data.name[0]
    subject_state = data.name[1]
    subject_scenario = data.name[2]
    index = [subject_id, subject_state, subject_scenario]
    start_timestamp = window_timestamps.loc[subject_id, subject_state, subject_scenario].iat[0]
    start_index = pd.MultiIndex.from_tuples([tuple(index + [start_timestamp])], names=GROUPING_COLUMNS + ['datetime'])
    end_timestamp = window_timestamps.loc[subject_id, subject_state, subject_scenario].iat[-1]
    end_index = pd.MultiIndex.from_tuples([tuple(index + [end_timestamp])], names=GROUPING_COLUMNS + ['datetime'])
    start_timestamp_row = pd.DataFrame([[np.nan] * len(data.columns)], columns=data.columns, index=start_index)
    end_timestamp_row = pd.DataFrame([[np.nan] * len(data.columns)], columns=data.columns, index=end_index)
    start_data = pd.concat([start_timestamp_row, data, end_timestamp_row], axis=0)
    resampled = start_data.resample('1S', level='datetime', origin='start').mean()
    rolling_window = resampled.rolling(
            str(window_size) + 'S', on=resampled.index.get_level_values('datetime'), closed='both'
        )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rolling_window_count = rolling_window.count()
        rolling_window_ratio = (rolling_window.sum() / window_size)['duration'].rename(event + '_event_ratio')
        rolling_window_mean = rolling_window.mean().add_prefix(event + '_event_').add_suffix('-mean')
        rolling_window_std = rolling_window.std(ddof=0).add_prefix(event + '_event_').add_suffix('-std')
        rolling_window_q5 = rolling_window.quantile(0.05).add_prefix(event + '_event_').add_suffix('-q5')
        rolling_window_q95 = rolling_window.quantile(0.95).add_prefix(event + '_event_').add_suffix('-q95')
        rolling_window_skew = rolling_window.skew().add_prefix(event + '_event_').add_suffix('-skewness')
        rolling_window_kurt = rolling_window.kurt().add_prefix(event + '_event_').add_suffix('-kurtosis')
        result = pd.concat(
            [
                rolling_window_count['duration'].rename(event + '_event_count'), 
                rolling_window_ratio.fillna(0), 
                rolling_window_mean.where(rolling_window_count.any(axis=1), 0), 
                rolling_window_std.where(rolling_window_count.any(axis=1), 0), 
                rolling_window_q5.where(rolling_window_count.any(axis=1), 0), 
                rolling_window_q95.where(rolling_window_count.any(axis=1), 0), 
                rolling_window_skew.where(rolling_window_count.any(axis=1), 0), 
                rolling_window_kurt.where(rolling_window_count.any(axis=1), 0)
            ], 
            axis=1
        )
        return result
