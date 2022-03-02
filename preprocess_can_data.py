import pandas as pd
import numpy as np
import glob
import re
import datetime
import pytz
import os
from event_functions import brake_to_gas, calculate_event_stats, distance_covered, gas_to_brake, get_overtaking_events, get_road_sign_events, get_turning_events
from id_extraction import get_distance_based_path_and_segment_ids, get_path_and_segment_ids


def filter_can_data_engine_on(can_data, timestamps):
    whole_trip_timestamp = timestamps[timestamps['reason'] == 'trip_start_end_times'].iloc[0]
    can_data_filtered = can_data[(can_data['timestamp'] > whole_trip_timestamp['start_time']) &
        (can_data['timestamp'] < whole_trip_timestamp['end_time'])]

    assert len(can_data_filtered[(can_data_filtered['timestamp'] < whole_trip_timestamp['start_time'])
        & (can_data_filtered['timestamp'] > whole_trip_timestamp['end_time'])].index) == 0

    engine_in_trip_off_timestamps = timestamps[timestamps['reason'] == 'engine_in_trip_off']

    for start_time, end_time in engine_in_trip_off_timestamps[['start_time', 'end_time']].to_numpy():
        can_data_filtered = can_data_filtered[(can_data_filtered['timestamp'] < start_time) |
            (can_data_filtered['timestamp'] > end_time)]

        assert len(can_data_filtered[(can_data_filtered['timestamp']> start_time)
            & (can_data_filtered['timestamp'] < end_time)].index) == 0

    return can_data_filtered.copy()


def split_signal_in_right_left(df, signal):
    df[signal + "_right"] = df.loc[df[signal] < 0, signal].abs()
    df[signal + "_right"] = df[signal + "_right"].fillna(0)
    df[signal + "_left"] = df.loc[df[signal] > 0, signal]
    df[signal + "_left"] = df[signal + "_right"].fillna(0)

    return df


def calculate_lane_pos(lanes_df, segment_id, latpos):
    lanes_in_segment = lanes_df.loc[lanes_df['segment_id'] == segment_id]
    if lanes_in_segment.empty:
        return -1, np.nan, np.nan, np.nan
    lanes_center = []
    lanes_left_edge = []
    lanes_right_edge = []
    prev_width = 0
    for _, row in lanes_in_segment.iterrows():
        lanes_center.append(row['LaneWidth'] / 2.0 + prev_width)
        lanes_left_edge.append(row['LaneWidth'] + prev_width)
        lanes_right_edge.append(prev_width)
        prev_width = row['LaneWidth']
    deviation = [np.abs(latpos - (c - lanes_center[0])) for c in lanes_center]
    lane_number = np.argmin(deviation)
    lane_position = abs(latpos - (lanes_center[lane_number] - lanes_center[0]))
    lane_distance_left_edge = (lanes_left_edge[lane_number] - lanes_center[0]) - latpos
    lane_distance_right_edge = latpos - (lanes_right_edge[lane_number] - lanes_center[0])
    return lane_number, lane_position, lane_distance_left_edge, lane_distance_right_edge


def do_derivation_of_signals(df, signals, suffix, frequency_hz=None, replace_suffix=None):
    if frequency_hz is None:
        time_delta = (df['timestamp'].iloc[1] - df['timestamp'].iloc[0]).total_seconds()
    else:
        time_delta = 1 / frequency_hz
    for signal in signals:
        if replace_suffix is not None and replace_suffix in signal:
            signal_new = signal.replace(replace_suffix, "")
        else:
            signal_new = signal
        df.loc[:, signal_new + suffix] = df[signal].diff() / time_delta
        df.loc[:, signal_new + suffix] = df[signal_new + suffix].fillna(0)

    return df


def get_lane_switching(data, direction=''):
    def calc_lane_switching(row):
        # check 5 seconds after lane crossing started if the lane number changes exactly once to indicate intended lane switching
        counts = data.iloc[row.name:row.name+150]['lane_number'].diff().abs().value_counts()
        return (counts[1] == 1).astype(int) if 1 in counts else 0
    lane_switching = data[data['lane_crossing'+direction] == 1].apply(calc_lane_switching, axis=1)
    lane_switching = lane_switching.reindex(list(range(data.index.min(), data.index.max()+1)), fill_value=0)
    return lane_switching


def do_preprocessing(full_study, overwrite, data_freq=30):
    if glob.glob('out/can_data.parquet') and not overwrite:
        return

    if os.path.exists('out/error_digits.txt'):
        os.remove('out/error_digits.txt')
    
    CAR_WIDTH = 1.75

    CAN_COLUMNS = ['interval', 'steer', 'latpos', 'gas', 'brake', 'clutch', 'Thw', 'velocity', 'acc', 'latvel', 'dtoint', 'indicator',
               'heading', 'SpeedDif', 'LaneDirection', 'SteerError', 'SteerSpeed', 'Ttc', 'TtcOpp', 'LeftDis',
               'RightDis', 'AheadDis', 'traflight', 'handbrake', 'engine']

    SIGNALS_WITH_POSITIVE_AND_NEGATIVE_VALUES = ["latpos", "steer", "latvel", "SteerSpeed", "SteerError"]

    SIGNALS_DERIVE_VELOCITY = ['gas', 'brake']
    SIGNALS_DERIVE_ACCELERATION = ['gas_vel', 'brake_vel', 'latvel', 'SteerSpeed', 'SpeedDif']
    SIGNALS_DERIVE_JERK = ['gas_acc', 'brake_acc', 'latvel_acc', 'SteerSpeed_acc', 'acc']

    gas_event_data = []
    brake_to_gas_event_data = []
    brake_event_data = []
    gas_to_brake_event_data = []
    overtaking_event_data = []
    turning_event_data = []
    road_sign_event_data = []
    data = []

    if full_study:
        DATA_FOLDER = "/adar/drive/study"
        CAN_COLUMNS = CAN_COLUMNS[:22] + ["ypos", "xpos", "YawRate"] + CAN_COLUMNS[22:]
        SIGNALS_WITH_POSITIVE_AND_NEGATIVE_VALUES = SIGNALS_WITH_POSITIVE_AND_NEGATIVE_VALUES + ["YawRate"]
        SIGNALS_DERIVE_ACCELERATION = SIGNALS_DERIVE_ACCELERATION + ["YawRate"]
        SIGNALS_DERIVE_JERK = SIGNALS_DERIVE_JERK + ["YawRate_acc"]

        subject_folders = sorted(glob.glob(DATA_FOLDER + '/*_sober') + glob.glob(DATA_FOLDER + '/*_above') + glob.glob(
            DATA_FOLDER + '/*_below'))
        subject_folders = [f for f in subject_folders if 'audio' not in f]
    else:
        DATA_FOLDER = "/adar/drive/pilot"
        subject_folders = sorted(glob.glob(DATA_FOLDER + '/*_drunk') + glob.glob(DATA_FOLDER + '/*_sober'))

    if not glob.glob(DATA_FOLDER):
        print('Data folder does not exist')
        return

    subject_re = re.compile('d-([0-9]+)')
    scenario_re = re.compile(r's\d{2}[abcde]-(\w*)Exp.dat$')
    timestamp_re = re.compile(r'(\d{4})-(\d{2})-(\d{2})--(\d{2})-(\d{2})-(\d{2}).flv')

    lanes_df = pd.read_csv('out/scenario_information.csv')

    road_signs_df = pd.read_csv('out/road_sign_information.csv')

    for subject in subject_folders:
        subject_id_match = subject_re.search(subject)
        if not subject_id_match:
            continue
        subject_id = subject_id_match.group(1)

        if "drunk" in subject:
            state = "drunk"
        elif "above" in subject:
            state = "above"
        elif "below" in subject:
            state = "below"
        else:
            state = "sober"

        dimensions = dict()
        if not (subject_id == '018' and state == 'sober'):
            dimensions['x'] = 1022
            dimensions['y'] = 1723
            dimensions['height'] = 76
            dimensions['width'] = 50
            dimensions['path_frame'] = 21
            dimensions['segment_frame'] = 29
        else:
            dimensions['x'] = 2624
            dimensions['y'] = 1696
            dimensions['height'] = 52
            dimensions['width'] = 35
            dimensions['path_frame'] = 15
            dimensions['segment_frame'] = 20

        timestamp_file = glob.glob(subject + '/timestamps.csv')
        timestamps = pd.read_csv(timestamp_file[0], sep=',', index_col=0, skiprows=0,
                            parse_dates=['start_time', 'end_time'])

        videos = sorted(glob.glob(subject + '/obs-videos/*[!_cropped].flv'))
        video = None
        if subject_id == '034' and state == 'sober':
            video = videos[1]
        else:
            video = videos[0]

        match = timestamp_re.search(video.split('/')[-1])
        if match:
            year = int(match.group(1))
            month = int(match.group(2))
            day = int(match.group(3))
            hour = int(match.group(4))
            minute = int(match.group(5))
            second = int(match.group(6))
            video_timestamp = datetime.datetime(year, month, day, hour, minute, second)
            tz = pytz.timezone('Europe/Zurich')
            video_timestamp = tz.localize(video_timestamp)

        for can_file in sorted(glob.glob(subject + '/simulator/*.dat')):
            scenario_re_match = scenario_re.search(can_file)

            if not scenario_re_match:
                continue
            scenario = scenario_re_match.group(1).lower()

            distance_based = False
            if ((subject_id == '024' and state == 'below')
            or (subject_id == '005' and state == 'above' and scenario == 'rural')
            or (subject_id == '031' and state == 'below' and scenario == 'rural')
            or (subject_id == '018' and state == 'sober' and scenario == 'highway')):
                distance_based = True

            print('subject id: {}, state: {}, scenario: {}'.format(subject_id, state, scenario))

            with open(can_file) as f:
                line = f.readline().replace('\n', '')
                line = line.replace('.   ', '.000').replace('.  ', '.00').replace('. ', '.0')
                simulator_timestamp = datetime.datetime.strptime(line, '%Y-%m-%d %H:%M:%S.%f')

            can_data = pd.read_csv(can_file, header=None, index_col=None, names=CAN_COLUMNS, delim_whitespace=True,
                skiprows=2, na_values='9999.00')
            can_data.loc[:, 'interval'] = [pd.to_datetime(simulator_timestamp + datetime.timedelta(milliseconds=interval * 1000))
                .tz_localize('Europe/Zurich') for interval in can_data['interval']]
            can_data.rename(columns={'interval': 'timestamp'}, inplace=True)

            timestamps_scenario = timestamps[timestamps['scenario'] == scenario]

            can_data_filtered = filter_can_data_engine_on(can_data, timestamps_scenario)

            can_data_filtered.insert(0, 'subject_id', subject_id)
            can_data_filtered.insert(1, 'subject_state', state)
            can_data_filtered.insert(2, 'subject_scenario', scenario)
            can_data_filtered.reset_index(drop=True, inplace=True)

            can_data_filtered.loc[:, "indicator_right"] = (can_data_filtered["indicator"] == 1).astype(int)
            can_data_filtered.loc[:, "indicator_left"] = (can_data_filtered["indicator"] == 2).astype(int)
            can_data_filtered.drop(["indicator"], axis=1, inplace=True)

            lanes_for_scenario = lanes_df.loc[lanes_df['scenario'] == scenario]
            if not distance_based:
                path_ids, segment_ids = get_path_and_segment_ids(video, dimensions, can_data_filtered['timestamp'], video_timestamp, data_freq)
                can_data_filtered.loc[:, 'path_id'] = path_ids
                can_data_filtered.loc[:, 'segment_id'] = segment_ids
            else:
                path_ids, segment_ids = get_distance_based_path_and_segment_ids(lanes_for_scenario, can_data_filtered['xpos'], can_data_filtered['ypos'])
                can_data_filtered.loc[:, 'path_id'] = path_ids
                can_data_filtered.loc[:, 'segment_id'] = segment_ids

            lane_info = [calculate_lane_pos(lanes_for_scenario, segment_id, latpos)
                        for segment_id, latpos in zip(can_data_filtered['segment_id'], can_data_filtered['latpos'])]
            lane_info = np.array(lane_info)

            can_data_filtered.loc[:, 'lane_number'] = lane_info[:, 0]
            can_data_filtered.loc[:, 'lane_position'] = lane_info[:, 1]
            can_data_filtered.loc[:, 'lane_distance_left_edge'] = lane_info[:, 2]
            can_data_filtered.loc[:, 'lane_distance_right_edge'] = lane_info[:, 3]
            can_data_filtered.loc[:, 'is_crossing_lane'] = ((can_data_filtered['lane_distance_left_edge'] < (CAR_WIDTH / 2.0))
                                                                | (can_data_filtered['lane_distance_right_edge'] < (CAR_WIDTH / 2.0))).astype(int)
            can_data_filtered.loc[:, 'lane_crossing'] = can_data_filtered['is_crossing_lane'].diff().abs()
            can_data_filtered.loc[:, 'lane_switching'] = get_lane_switching(can_data_filtered)
            can_data_filtered.loc[:, 'is_crossing_lane_left'] = (can_data_filtered['lane_distance_left_edge'] < (CAR_WIDTH / 2.0)).astype(int)
            can_data_filtered.loc[:, 'is_crossing_lane_right'] = (can_data_filtered['lane_distance_right_edge'] < (CAR_WIDTH / 2.0)).astype(int)
            can_data_filtered.loc[:, 'lane_crossing_left'] = (can_data_filtered['is_crossing_lane_left'].diff() == 1).astype(int)
            can_data_filtered.loc[:, 'lane_crossing_right'] = (can_data_filtered['is_crossing_lane_right'].diff() == 1).astype(int)
            if scenario != 'highway':
                can_data_filtered.loc[:, 'opp_lane_switching'] = get_lane_switching(can_data_filtered, '_left')
            else:
                can_data_filtered.loc[:, 'opp_lane_switching'] = 0

            can_data_filtered.loc[:, 'Dhw'] = can_data_filtered['Thw'] * can_data_filtered['velocity']

            can_data_filtered.loc[:, 'speed_limit_exceeded'] = (can_data_filtered['SpeedDif'] > 0).astype(int)

            can_data_filtered = do_derivation_of_signals(can_data_filtered, SIGNALS_DERIVE_VELOCITY, '_vel', data_freq)
            can_data_filtered = do_derivation_of_signals(can_data_filtered, SIGNALS_DERIVE_ACCELERATION, '_acc', data_freq, '_vel')
            can_data_filtered = do_derivation_of_signals(can_data_filtered, SIGNALS_DERIVE_JERK, '_jerk', data_freq, '_acc')

            data.append(can_data_filtered)

            positive_brake_events = (can_data_filtered[can_data_filtered['brake'] > 0]
                .groupby((can_data_filtered['brake'] == 0).cumsum(), as_index=False)
                .filter(lambda x: (x['gas'] == 0).all())
                .groupby((can_data_filtered['brake'] == 0).cumsum(), as_index=False))
            zero_gas_events = can_data_filtered[can_data_filtered['gas'] == 0].groupby((can_data_filtered['gas'] > 0).cumsum(), as_index=False)
            gas_to_brake_event = zero_gas_events.apply(gas_to_brake)
            gas_to_brake_event.insert(0, 'subject_id', subject_id)
            gas_to_brake_event.insert(1, 'subject_state', state)
            gas_to_brake_event.insert(2, 'subject_scenario', scenario)
            brake_dist_covered = positive_brake_events.apply(distance_covered)
            brake_events_stats = calculate_event_stats(positive_brake_events, 'brake')
            brake_events_stats.columns = brake_events_stats.columns.map('_'.join)
            brake_events_stats = pd.concat((brake_events_stats, brake_dist_covered), axis=1)
            brake_events_stats.insert(0, 'subject_id', subject_id)
            brake_events_stats.insert(1, 'subject_state', state)
            brake_events_stats.insert(2, 'subject_scenario', scenario)
            brake_events_stats.reset_index(drop=True, inplace=True)

            brake_event_data.append(brake_events_stats)
            gas_to_brake_event_data.append(gas_to_brake_event)

            positive_gas_events = (can_data_filtered[can_data_filtered['gas'] > 0]
                .groupby((can_data_filtered['gas'] == 0).cumsum(), as_index=False)
                .filter(lambda x: (x['brake'] == 0).all())
                .groupby((can_data_filtered['gas'] == 0).cumsum(), as_index=False))
            zero_brake_events = can_data_filtered[can_data_filtered['brake'] == 0].groupby((can_data_filtered['brake'] > 0).cumsum(), as_index=False)
            brake_to_gas_event = zero_brake_events.apply(brake_to_gas)
            brake_to_gas_event.insert(0, 'subject_id', subject_id)
            brake_to_gas_event.insert(1, 'subject_state', state)
            brake_to_gas_event.insert(2, 'subject_scenario', scenario)
            gas_dist_covered = positive_gas_events.apply(distance_covered)
            gas_events_stats = calculate_event_stats(positive_gas_events, 'gas')
            gas_events_stats.columns = gas_events_stats.columns.map('_'.join)
            gas_events_stats = pd.concat((gas_events_stats, gas_dist_covered), axis=1)
            gas_events_stats.insert(0, 'subject_id', subject_id)
            gas_events_stats.insert(1, 'subject_state', state)
            gas_events_stats.insert(2, 'subject_scenario', scenario)
            gas_events_stats.reset_index(drop=True, inplace=True)

            gas_event_data.append(gas_events_stats)
            brake_to_gas_event_data.append(brake_to_gas_event)

            lane_zero_to_one = can_data_filtered[can_data_filtered['lane_number'] == 1].groupby((can_data_filtered['lane_number'] == 0).cumsum(), as_index=False)
            overtaking_events_stats = get_overtaking_events(can_data_filtered, lane_zero_to_one)
            if scenario == 'highway':
                lane_one_to_two = can_data_filtered[can_data_filtered['lane_number'] == 2].groupby((can_data_filtered['lane_number'] == 1).cumsum(), as_index=False)
                overtaking_events_stats = pd.concat((overtaking_events_stats, get_overtaking_events(can_data_filtered, lane_one_to_two)), ignore_index=True)
            else:
                turning_events_stats = get_turning_events(can_data_filtered)
                turning_events_stats.insert(0, 'subject_id', subject_id)
                turning_events_stats.insert(1, 'subject_state', state)
                turning_events_stats.insert(2, 'subject_scenario', scenario)
                turning_event_data.append(turning_events_stats)
            
            overtaking_events_stats.insert(0, 'subject_id', subject_id)
            overtaking_events_stats.insert(1, 'subject_state', state)
            overtaking_events_stats.insert(2, 'subject_scenario', scenario)
            overtaking_event_data.append(overtaking_events_stats)

            SPEED_LIMIT_30 = 1
            SPEED_LIMIT_50 = 2
            SPEED_LIMIT_60 = 140
            SPEED_LIMIT_80 = 141
            SPEED_LIMIT_100 = 4
            SPEED_LIMIT_120 = 5
            RIGHT_OF_WAY = 20
            RIGHT_OF_WAY_LEFT = 21
            RIGHT_OF_WAY_RIGHT = 22
            STOP_SIGN = 24
            SPEED_BUMP = 94
            PED_CROSSING_WARNING = 107
            PED_CROSSING = 114

            signs_for_scenario = road_signs_df[road_signs_df['scenario'] == scenario]
            speed_limit_30 = signs_for_scenario[signs_for_scenario['signType'] == SPEED_LIMIT_30]
            speed_limit_50 = signs_for_scenario[signs_for_scenario['signType'] == SPEED_LIMIT_50]
            speed_limit_60 = signs_for_scenario[signs_for_scenario['signType'] == SPEED_LIMIT_60]
            speed_limit_80 = signs_for_scenario[signs_for_scenario['signType'] == SPEED_LIMIT_80]
            speed_limit_100 = signs_for_scenario[signs_for_scenario['signType'] == SPEED_LIMIT_100]
            speed_limit_120 = signs_for_scenario[signs_for_scenario['signType'] == SPEED_LIMIT_120]

            speed_limit_30_events_stats = get_road_sign_events(speed_limit_30, can_data_filtered, 'speed_limit_30', subject_id, state, scenario)
            speed_limit_50_events_stats = get_road_sign_events(speed_limit_50, can_data_filtered, 'speed_limit_50', subject_id, state, scenario)
            speed_limit_60_events_stats = get_road_sign_events(speed_limit_60, can_data_filtered, 'speed_limit_60', subject_id, state, scenario)
            speed_limit_80_events_stats = get_road_sign_events(speed_limit_80, can_data_filtered, 'speed_limit_80', subject_id, state, scenario)
            speed_limit_100_events_stats = get_road_sign_events(speed_limit_100, can_data_filtered, 'speed_limit_100', subject_id, state, scenario)
            speed_limit_120_events_stats = get_road_sign_events(speed_limit_120, can_data_filtered, 'speed_limit_120', subject_id, state, scenario)

            right_of_way = signs_for_scenario[(
                (signs_for_scenario['signType'] == RIGHT_OF_WAY)
                | (signs_for_scenario['signType'] == RIGHT_OF_WAY_LEFT)
                | (signs_for_scenario['signType'] == RIGHT_OF_WAY_RIGHT))]
            stop_signs = signs_for_scenario[signs_for_scenario['signType'] == STOP_SIGN]
            speed_bumps = signs_for_scenario[signs_for_scenario['signType'] == SPEED_BUMP]
            ped_crossing_warnings = signs_for_scenario[signs_for_scenario['signType'] == PED_CROSSING_WARNING]
            ped_crossings = signs_for_scenario[signs_for_scenario['signType'] == PED_CROSSING]

            right_of_way_events_stats = get_road_sign_events(right_of_way, can_data_filtered, 'right_of_way', subject_id, state, scenario)
            stop_sign_events_stats = get_road_sign_events(stop_signs, can_data_filtered, 'stop', subject_id, state, scenario)
            speed_bumps_events_stats = get_road_sign_events(speed_bumps, can_data_filtered, 'speed_bump', subject_id, state, scenario)
            ped_crossing_warning_events_stats = get_road_sign_events(ped_crossing_warnings, can_data_filtered, 'pedestrian_crossing_warning', subject_id, state, scenario)
            ped_crossings_events_stats = get_road_sign_events(ped_crossings, can_data_filtered, 'pedestrian_crossing', subject_id, state, scenario)

            road_sign_events_stats = pd.concat((
                speed_limit_30_events_stats,
                speed_limit_50_events_stats,
                speed_limit_60_events_stats,
                speed_limit_80_events_stats,
                speed_limit_100_events_stats,
                speed_limit_120_events_stats,
                right_of_way_events_stats,
                stop_sign_events_stats,
                speed_bumps_events_stats,
                ped_crossing_warning_events_stats,
                ped_crossings_events_stats
                ))
            road_sign_event_data.append(road_sign_events_stats)

            

    data = pd.concat(data)
    data.to_parquet("out/can_data.parquet")

    brake_event_data = pd.concat(brake_event_data)
    brake_event_data.to_parquet('out/can_data_brake_events.parquet')

    gas_to_brake_event_data = pd.concat(gas_to_brake_event_data)
    gas_to_brake_event_data.to_parquet('out/can_data_gas_to_brake_events.parquet')

    gas_event_data = pd.concat(gas_event_data)
    gas_event_data.to_parquet('out/can_data_gas_events.parquet')

    brake_to_gas_event_data = pd.concat(brake_to_gas_event_data)
    brake_to_gas_event_data.to_parquet('out/can_data_brake_to_gas_events.parquet')

    overtaking_event_data = pd.concat(overtaking_event_data)
    overtaking_event_data.to_parquet('out/can_data_overtaking_events.parquet')

    turning_event_data = pd.concat(turning_event_data)
    turning_event_data.to_parquet('out/can_data_turning_events.parquet')

    road_sign_event_data = pd.concat(road_sign_event_data)
    road_sign_event_data.to_parquet('out/can_data_road_sign_events.parquet')
