import pandas as pd
import numpy as np
import glob
import re
import datetime
import pytz
import cv2 as cv


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


def map_to_digit(segments):
    zero = np.array([1, 1, 1, 0, 1, 1, 1])
    one = np.array([0, 0, 1, 0, 0, 1, 0])
    two = np.array([1, 0, 1, 1, 1, 0, 1])
    three = np.array([1, 0, 1, 1, 0, 1, 1])
    four = np.array([0, 1, 1, 1, 0, 1, 0])
    five = np.array([1, 1, 0, 1, 0, 1, 1])
    six = np.array([1, 1, 0, 1, 1, 1, 1])
    seven = np.array([1, 0, 1, 0, 0, 1, 0])
    eight = np.array([1, 1, 1, 1, 1, 1, 1])
    nine = np.array([1, 1, 1, 1, 0, 1, 0])

    if np.array_equal(segments, zero):
        return '0'
    elif np.array_equal(segments, one):
        return '1'
    elif np.array_equal(segments, two):
        return '2'
    elif np.array_equal(segments, three):
        return '3'
    elif np.array_equal(segments, four):
        return '4'
    elif np.array_equal(segments, five):
        return '5'
    elif np.array_equal(segments, six):
        return '6'
    elif np.array_equal(segments, seven):
        return '7'
    elif np.array_equal(segments, eight):
        return '8'
    elif np.array_equal(segments, nine):
        return '9'


def get_segment_states(digits):
    segments_states = []
    for i in range(len(digits)):
        digit = digits[i]
        contours, _ = cv.findContours(digit, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            continue
        full_contours = np.concatenate(contours)
        x, y, w, h = cv.boundingRect(full_contours)
        # fix for digit 1 and 3 (bounding box too thin, extend towards left)
        if w <= 8:
            x = x - (9 - w)
            w = 9
        digit_rect = digit[y:y + h, x:x + w]
        (digit_h, digit_w) = digit_rect.shape
        (segment_w, segment_h) = (int(digit_w * 0.25), int(digit_h * 0.15))
        segment_h_center = int(digit_h * 0.10)
        segments = [
            ((0, 0), (w, segment_h)),	# top
            ((0, 0), (segment_w, h // 2)),	# top-left
            ((w - segment_w, 0), (w, h // 2)),	# top-right
            ((0, (h // 2) - segment_h_center) , (w, (h // 2) + segment_h_center)), # center
            ((0, h // 2), (segment_w, h)),	# bottom-left
            ((w - segment_w, h // 2), (w, h)),	# bottom-right
            ((0, h - segment_h), (w, h))	# bottom
        ]
        segment_state = np.array([0] * len(segments))
        for j, ((xA, yA), (xB, yB)) in enumerate(segments):
            segment = digit_rect[yA:yB, xA:xB]
            total = cv.countNonZero(segment)
            area = (xB - xA) * (yB - yA)
            if total / float(area) > 0.5:
                segment_state[j]= 1
        segments_states.append(segment_state)
    
    return segments_states


def get_path_and_segment_ids(video, data_timestamps, video_timestamp):
    can_data_timestamps_ms = (data_timestamps - video_timestamp) / datetime.timedelta(milliseconds=1)
    cap = cv.VideoCapture(video)
    path_ids = []
    segment_ids = []
    for timestamp_ms in can_data_timestamps_ms:
        is_set = cap.set(cv.CAP_PROP_POS_MSEC, timestamp_ms)
        if is_set:
            success, frame = cap.read()
            if success:
                path_img = frame[1723:1743, 1022:1098, :]
                imgray = cv.cvtColor(path_img, cv.COLOR_BGR2GRAY)
                _, thresh = cv.threshold(imgray, 127, 255, cv.THRESH_BINARY)
                digits = np.array_split(thresh, 5, axis=1)
                path_id = ''
                for segment_state in get_segment_states(digits):
                    path_id += map_to_digit(segment_state)
                path_id = int(path_id)
                path_ids.append(path_id)

                segment_img = frame[1752:1772, 1022:1098, :]
                imgray = cv.cvtColor(segment_img, cv.COLOR_BGR2GRAY)
                _, thresh = cv.threshold(imgray, 127, 255, cv.THRESH_BINARY)
                digits = np.array_split(thresh, 5, axis=1)
                segment_id = ''
                for segment_state in get_segment_states(digits):
                    segment_id += map_to_digit(segment_state)
                segment_id = int(segment_id)
                segment_ids.append(segment_id)
            else:
                print('could not get frame')
        else:
            print('could not set video timestamp')

    return np.array(path_ids), np.array(segment_ids)


def calculate_lane_pos(lanes_df, segment_id, latpos):
    lanes_in_segment = lanes_df.loc[lanes_df['segment_id'] == segment_id]
    lanes_center = []
    prev_width = 0
    for _, row in lanes_in_segment.iterrows():
        lanes_center.append(row['LaneWidth'] / 2.0 + row['RightEdgeLineWidth'] + prev_width)
        prev_width = row['LaneWidth'] + row['RightEdgeLineWidth']
    deviation = [np.abs(latpos - c) for c in lanes_center]
    lane_number = np.argmin(deviation)
    lane_position = None
    if lane_number == 0:
        lane_position = latpos
    else:
        lane_position = latpos - (lanes_center[lane_number] - lanes_center[0])
    return lane_number, lane_position


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


def do_preprocessing(full_study, overwrite, data_freq=30):
    if glob.glob('out/can_data.parquet') and not overwrite:
        return

    CAN_COLUMNS = ['interval', 'steer', 'latpos', 'gas', 'brake', 'clutch', 'Thw', 'velocity', 'acc', 'latvel', 'dtoint', 'indicator',
               'heading', 'SpeedDif', 'LaneDirection', 'SteerError', 'SteerSpeed', 'Ttc', 'TtcOpp', 'LeftDis',
               'RightDis', 'AheadDis', 'traflight', 'handbrake', 'engine']

    SIGNALS_WITH_POSITIVE_AND_NEGATIVE_VALUES = ["latpos", "steer", "latvel", "SteerSpeed", "SteerError"]

    SIGNALS_DERIVE_VELOCITY = ['gas', 'brake']
    SIGNALS_DERIVE_ACCELERATION = ['gas_vel', 'brake_vel', 'latvel', 'SteerSpeed', 'SpeedDif']
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

        timestamp_file = glob.glob(subject + '/timestamps.csv')
        timestamps = pd.read_csv(timestamp_file[0], sep=',', index_col=0, skiprows=0,
                            parse_dates=['start_time', 'end_time'])

        video = glob.glob(subject + '/obs-videos/*.flv')[0]
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

            path_ids, segment_ids = get_path_and_segment_ids(video, can_data_filtered['timestamp'], video_timestamp)
            can_data_filtered.loc[:, 'path_id'] = path_ids
            can_data_filtered.loc[:, 'segment_id'] = segment_ids

            lanes_on_route = lanes_df.loc[lanes_df['scenario'] == scenario]
            lane_info = [calculate_lane_pos(lanes_on_route, segment_id, latpos)
                    for segment_id, latpos in zip(can_data_filtered['segment_id'], can_data_filtered['latpos'])]
            lane_info = np.array(lane_info)
            
            can_data_filtered.loc[:, 'lane_number'] = lane_info[:, 0]
            can_data_filtered.loc[:, 'lane_position'] = lane_info[:, 1]

            can_data_filtered.loc[:, 'Dhw'] = can_data_filtered['Thw'] * can_data_filtered['velocity']

            can_data_filtered = do_derivation_of_signals(can_data_filtered, SIGNALS_DERIVE_VELOCITY, '_vel', data_freq)
            can_data_filtered = do_derivation_of_signals(can_data_filtered, SIGNALS_DERIVE_ACCELERATION, '_acc', data_freq, '_vel')
            can_data_filtered = do_derivation_of_signals(can_data_filtered, SIGNALS_DERIVE_JERK, '_jerk', data_freq, '_acc')

            data.append(can_data_filtered)
    
    data = pd.concat(data)
    data.to_parquet("out/can_data.parquet")
