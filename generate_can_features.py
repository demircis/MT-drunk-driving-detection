import pandas as pd
from flirt.stats import get_stat_features

GROUPING_COLUMNS = ["subject_id", "subject_state", "subject_scenario"]

DRIVER_BEHAVIOR = ['steer', 'gas', 'brake', 'SteerSpeed', 'indicator_left', 'indicator_right',
                    'gas_vel', 'brake_vel', 'gas_acc', 'brake_acc', 'SteerSpeed_acc',
                    'gas_jerk', 'brake_jerk', 'SteerSpeed_jerk', 'SteerError']
VEHICLE_BEHAVIOR = ['velocity', 'acc', 'acc_jerk', 'latvel', 'YawRate', 'latvel_acc', 'latvel_jerk',
                    'YawRate_acc', 'YawRate_jerk']
RADAR = ["LaneDirection", "latpos", "Ttc", 'TtcOpp', "Thw"]
NAVI = ["SpeedDif"]

SIGNALS = {'driver_behavior': DRIVER_BEHAVIOR, 'vehicle_behavior': VEHICLE_BEHAVIOR, 'radar': RADAR, 'navi': NAVI}

def can_window_agg(data, window_length, data_freq=30, num_cores=0):
    feature_data = data.drop(columns=GROUPING_COLUMNS)
    feature = get_stat_features(feature_data, window_length, window_step_size=1,
                                data_frequency=data_freq, entropies=False, num_cores=num_cores)
    return feature

def store_can_features(window_sizes):

    can_data = pd.read_parquet("out/can_data.parquet")

    for key, signal in SIGNALS.items():
        for window_size in window_sizes:
            can_data_features = can_data.groupby(GROUPING_COLUMNS).apply(
                lambda x: can_window_agg(x.set_index('timestamp')[GROUPING_COLUMNS + signal], window_size)
                )
            
            can_data_features.to_parquet('out/can_data_features_{}_windowsize_{}s.parquet'.format(key, window_size))
        
