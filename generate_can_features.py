import pandas as pd

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
