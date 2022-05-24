import pandas as pd
import numpy as np
import glob
import re
import datetime
import pytz
import os
from event_functions import validate_start_end_indices
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


def calculate_lane_pos(lanes_df, segment_id, latpos, dtoint):
    lanes_in_segment = lanes_df.loc[lanes_df['segment_id'] == segment_id]
    if lanes_in_segment.empty or dtoint < 0:
        return np.nan, np.nan, np.nan, np.nan
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
        start, end = validate_start_end_indices(data, row.name, row.name+150)
        counts = data.loc[start:end]['lane_number'].diff().abs().value_counts()
        return (counts[1] == 1).astype(int) if 1 in counts else 0
    lane_switching = data[data['lane_crossing'+direction] == 1].apply(calc_lane_switching, axis=1)
    if lane_switching.empty:
        lane_switching = pd.Series(dtype=np.float64)
    lane_switching = lane_switching.reindex(data.index, fill_value=0)
    return lane_switching


def do_preprocessing(full_study, data_freq=30):
    if os.path.exists('out/error_digits.txt'):
        os.remove('out/error_digits.txt')
    
    CAR_WIDTH = 1.75

    CAN_COLUMNS = ['interval', 'steer', 'latpos', 'gas', 'brake', 'clutch', 'Thw', 'velocity', 'acc', 'latvel', 'dtoint', 'indicator',
               'heading', 'SpeedDif', 'LaneDirection', 'SteerError', 'SteerSpeed', 'Ttc', 'TtcOpp', 'LeftDis',
               'RightDis', 'AheadDis', 'traflight', 'handbrake', 'engine']

    SIGNALS_WITH_POSITIVE_AND_NEGATIVE_VALUES = ["latpos", "steer", "latvel", "SteerSpeed", "SteerError"]

    SIGNALS_DERIVE_VELOCITY = ['gas', 'brake']
    SIGNALS_DERIVE_ACCELERATION = ['gas_vel', 'brake_vel', 'latvel', 'SteerSpeed']
    SIGNALS_DERIVE_JERK = ['gas_acc', 'brake_acc', 'latvel_acc', 'SteerSpeed_acc', 'acc']

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

            angle_for_segment = lanes_for_scenario[['segment_id', 'angle']].drop_duplicates().set_index('segment_id')
            can_data_filtered.loc[:, 'turn_angle'] = [angle_for_segment.loc[segment_id, 'angle'] if segment_id in angle_for_segment.index else -1 for segment_id in can_data_filtered['segment_id']]
            
            lane_info = [calculate_lane_pos(lanes_for_scenario, segment_id, latpos, dtoint)
                        for segment_id, latpos, dtoint in zip(can_data_filtered['segment_id'], can_data_filtered['latpos'], can_data_filtered['dtoint'])]
            lane_info = pd.DataFrame(lane_info)

            can_data_filtered.loc[:, 'lane_number'] = lane_info.iloc[:, 0].interpolate(method='nearest').fillna(method='bfill').fillna(method='ffill')
            can_data_filtered.loc[:, 'lane_position'] = lane_info.iloc[:, 1].interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')
            can_data_filtered.loc[:, 'lane_distance_left_edge'] = lane_info.iloc[:, 2].interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')
            can_data_filtered.loc[:, 'lane_distance_right_edge'] = lane_info.iloc[:, 3].interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')
            can_data_filtered.loc[:, 'is_crossing_lane'] = ((can_data_filtered['lane_distance_left_edge'] < (CAR_WIDTH / 2.0))
                                                                | (can_data_filtered['lane_distance_right_edge'] < (CAR_WIDTH / 2.0))).astype(int)
            can_data_filtered.loc[:, 'lane_crossing'] = can_data_filtered['is_crossing_lane'].diff().abs().fillna(method='bfill')
            can_data_filtered.loc[:, 'lane_switching'] = get_lane_switching(can_data_filtered)
            can_data_filtered.loc[:, 'is_crossing_lane_left'] = (can_data_filtered['lane_distance_left_edge'] < (CAR_WIDTH / 2.0)).astype(int)
            can_data_filtered.loc[:, 'is_crossing_lane_right'] = (can_data_filtered['lane_distance_right_edge'] < (CAR_WIDTH / 2.0)).astype(int)
            can_data_filtered.loc[:, 'lane_crossing_left'] = (can_data_filtered['is_crossing_lane_left'].diff() == 1).astype(int).fillna(method='bfill')
            can_data_filtered.loc[:, 'lane_crossing_right'] = (can_data_filtered['is_crossing_lane_right'].diff() == 1).astype(int).fillna(method='bfill')
            if scenario != 'highway':
                can_data_filtered.loc[:, 'opp_lane_switching'] = get_lane_switching(can_data_filtered, '_left')
            else:
                can_data_filtered.loc[:, 'opp_lane_switching'] = 0

            can_data_filtered.loc[:, 'Dhw'] = can_data_filtered['Thw'] * can_data_filtered['velocity']

            can_data_filtered.loc[:, 'speed_limit_exceeded'] = (can_data_filtered['SpeedDif'] > 0).astype(int)

            can_data_filtered = do_derivation_of_signals(can_data_filtered, SIGNALS_DERIVE_VELOCITY, '_vel', data_freq)
            can_data_filtered = do_derivation_of_signals(can_data_filtered, SIGNALS_DERIVE_ACCELERATION, '_acc', data_freq, '_vel')
            can_data_filtered = do_derivation_of_signals(can_data_filtered, SIGNALS_DERIVE_JERK, '_jerk', data_freq, '_acc')

            can_data_filtered.insert(0, 'subject_id', subject_id)
            can_data_filtered.insert(1, 'subject_state', state)
            can_data_filtered.insert(2, 'subject_scenario', scenario)
            data.append(can_data_filtered)

    data = pd.concat(data)
    data.to_parquet("out/can_data.parquet")
