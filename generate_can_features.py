import pandas as pd
from flirt.stats import get_stat_features

GROUPING_COLUMNS = ["subject_id", "subject_state", "subject_scenario"]
SELECTED_FEATURES = ['steer', 'latpos', 'gas', 'brake', 'clutch', 'Thw', 'velocity', 'acc', 'latvel',
        'heading', 'SpeedDif', 'LaneDirection', 'SteerError',
        'SteerSpeed', 'Ttc', 'TtcOpp', 'LeftDis', 'RightDis', 'AheadDis', 'YawRate',
        'indicator_right', 'indicator_left', 'gas_slope', 'brake_slope',
        'latvel_acc', 'SteerSpeed_acc', 'YawRate_acc', 'latvel_jerk',
        'SteerSpeed_jerk', 'YawRate_jerk']

def can_window_agg(data, window_length, data_freq=30, num_cores=0):
    feature_data = data.drop(columns=GROUPING_COLUMNS)
    feature = get_stat_features(feature_data, window_length, 1, data_frequency=data_freq, entropies=False, num_cores=num_cores)
    return feature

def store_can_features(window_sizes):

    can_data = pd.read_parquet("out/can_data.parquet")

    for window_size in window_sizes:
        can_data.groupby(GROUPING_COLUMNS).apply(lambda x: can_window_agg(x[GROUPING_COLUMNS + SELECTED_FEATURES], window_size))
        print(can_data[(can_data['subject_id'] == '006') & (can_data['subject_state'] == 'sober') & (can_data['subject_scenario'] == 'rural')])
        
