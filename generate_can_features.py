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
NAVI = ['dtoint', 'SpeedDif', 'speed_limit_exceeded']

SIGNALS = {'driver_behavior': DRIVER_BEHAVIOR, 'vehicle_behavior': VEHICLE_BEHAVIOR, 'radar': RADAR, 'navi': NAVI}

def can_window_agg(data, window_length, data_freq=30, num_cores=0):
    feature_data = data.drop(columns=GROUPING_COLUMNS)
    feature = get_features(feature_data, window_length, num_cores=num_cores,
                                step_size=data_freq)
    return feature


def store_can_data_features(window_sizes):

    can_data = pd.read_parquet('out/can_data.parquet')

    for key, signal in SIGNALS.items():
        for window_size in window_sizes:
            can_data_features = can_data.groupby(GROUPING_COLUMNS).apply(
                lambda x: can_window_agg(x.set_index('timestamp')[GROUPING_COLUMNS + signal], window_size, num_cores=10)
                )
            
            can_data_features.to_parquet('out/can_data_features_{}_windowsize_{}s.parquet'.format(key, window_size))


def store_can_event_features():

    EVENTS = ['brake', 'brake_to_gas', 'gas', 'gas_to_brake', 'overtaking', 'road_sign', 'turning']

    can_events_features = []
    for e in EVENTS:
        can_events_features.append(pd.read_parquet('out/can_data_{}_events.parquet'.format(e)))
    can_events_features = pd.concat(can_events_features, axis=0)

    #can_events_features.groupby(GROUPING_COLUMNS).to_parquet()
