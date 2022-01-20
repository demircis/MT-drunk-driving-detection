import pandas as pd
import numpy as np
import glob
import re
import datetime
import pytz
import cv2 as cv
import ffmpeg
import os


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
    else:
        return None


def get_segment_states(digits):
    segments_states = []
    for digit in digits:
        contours, _ = cv.findContours(digit, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            continue
        full_contours = np.concatenate(contours)
        x, y, w, h = cv.boundingRect(full_contours)
        # extend bounding box to the left (fix for certain digits)
        standard_w = 9
        if w < standard_w:
            x = x - (standard_w - w)
            w = standard_w
        digit_rect = digit[y:y + h, x:x + w]
        (segment_w, segment_h) = (3, 3)
        segment_h_center = 2
        segments = [
            ((1, 0), (w-1, segment_h)),	# top
            ((0, 1), (segment_w, h // 2)),	# top-left
            ((w - segment_w, 1), (w, h // 2)),	# top-right
            ((1, (h // 2) - segment_h_center) , (w-1, (h // 2) + segment_h_center)), # center
            ((0, h // 2), (segment_w, h-1)),	# bottom-left
            ((w - segment_w, h // 2), (w, h-1)),	# bottom-right
            ((1, h - segment_h), (w-1, h))	# bottom
        ]
        segment_state = np.array([0] * len(segments))
        for j, ((xA, yA), (xB, yB)) in enumerate(segments):
            segment = digit_rect[yA:yB, xA:xB]
            total = cv.countNonZero(segment)
            area = (xB - xA) * (yB - yA)
            if area == 0:
                segment_state = np.array([0] * len(segments))
                break
            if total / float(area) >= 0.5:
                segment_state[j]= 1
        segments_states.append(segment_state)
    
    return segments_states


def extract_id(img):
    imhsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    mask = cv.inRange(imhsv, (0, 50, 0), (179, 255, 255))
    img_no_artifacts = cv.bitwise_and(img, img, mask=mask)
    imgray = cv.cvtColor(img_no_artifacts, cv.COLOR_BGR2GRAY)
    _, thresh = cv.threshold(imgray, 127, 255, cv.THRESH_BINARY)
    digits = np.array_split(thresh, 5, axis=1)
    extracted_id = ''
    for segment_state in get_segment_states(digits):
        extracted_id += map_to_digit(segment_state)
    extracted_id = int(extracted_id) if extracted_id != '' else None
    return extracted_id


def get_ids_for_indices(path_ids, segment_ids, cropped_video, dimensions, timestamps, indices):
    cap = cv.VideoCapture(cropped_video)
    for ind, timestamp_ms in zip(indices, timestamps[indices]):
        is_set = cap.set(cv.CAP_PROP_POS_MSEC, timestamp_ms)
        success, frame = cap.read()
        if is_set and success:
            end = dimensions['path_frame']
            start = dimensions['segment_frame']
            path_img = frame[:end, :, :]
            segment_img = frame[start:, :, :]
            try:
                path_ids[ind] = extract_id(path_img)
                segment_ids[ind] = extract_id(segment_img)
            except TypeError:
                subject_folder = cropped_video.split('/')[4]
                if not os.path.exists('out/{}'.format(subject_folder)):
                    os.makedirs('out/{}'.format(subject_folder))
                cv.imwrite('out/{}/error_path_{}.jpg'.format(subject_folder, timestamp_ms), path_img)
                cv.imwrite('out/{}/error_segment_{}.jpg'.format(subject_folder, timestamp_ms), segment_img)
                f = open('out/error_digits.txt', 'a')
                f.write('video: {}, timestamp (ms): {}, index: {}\n'.format(cropped_video, timestamp_ms, ind))
                f.close()
        else:
            print('could not get frame at timestamp')
            break
    
    cap.release()
    return path_ids, segment_ids


def crop_video(video, crop_dimensions):
    (
    ffmpeg
    .input(video)
    .crop(crop_dimensions['x'], crop_dimensions['y'], crop_dimensions['height'], crop_dimensions['width'])
    .output(video[:-4] + '_cropped.flv', vcodec='libx264', acodec='copy', preset='ultrafast')
    .run()
    )


def get_path_and_segment_ids(video, dimensions, data_timestamps, video_timestamp, data_freq):
    can_data_timestamps_ms = ((data_timestamps - video_timestamp) / datetime.timedelta(milliseconds=1)).to_numpy()
    nr_timestamps = len(can_data_timestamps_ms)
    path_ids = np.array([None] * nr_timestamps)
    segment_ids = np.array([None] * nr_timestamps)

    sampling_indices = np.arange(0, nr_timestamps, data_freq * 5)
    last_index = nr_timestamps-1
    if sampling_indices[-1] != last_index:
        sampling_indices = np.append(sampling_indices, last_index)

    cropped_video = video[:-4] + '_cropped.flv'
    assert(cropped_video != video)

    if not os.path.exists(cropped_video):
        crop_video(video, dimensions)
    
    path_ids, segment_ids = get_ids_for_indices(path_ids, segment_ids, cropped_video, dimensions, can_data_timestamps_ms, sampling_indices)
    
    repeat = 0
    prev_none = -1
    while len(segment_ids[segment_ids == None]) != 0:
        new_indices = []
        for i in range(len(sampling_indices)-1):
            left = sampling_indices[i]
            right = sampling_indices[i+1]
            if path_ids[left] == path_ids[right]:
                path_ids[left:right] = path_ids[left]

            if (segment_ids[left] == segment_ids[right]) and not (segment_ids[left] == None and segment_ids[right] == None):
                segment_ids[left:right] = segment_ids[left] if segment_ids[left] != None else segment_ids[right]
            else:
                new_indices.append(left+(right-left)//2)

        new_indices = np.array(new_indices)
        sampling_indices = np.concatenate((sampling_indices, new_indices))
        sampling_indices = np.sort(sampling_indices)
        path_ids, segment_ids = get_ids_for_indices(path_ids, segment_ids, cropped_video, dimensions, can_data_timestamps_ms, new_indices)

        if len(segment_ids[segment_ids == None]) == prev_none:
            repeat += 1
            sampling_indices = sampling_indices[segment_ids[sampling_indices] != None]
        prev_none = len(segment_ids[segment_ids == None])
        
        if repeat == 3:
            break

    return path_ids, segment_ids


def calculate_lane_pos(lanes_df, segment_id, latpos):
    lanes_in_segment = lanes_df.loc[lanes_df['segment_id'] == segment_id]
    if lanes_in_segment.empty:
        return -1, np.nan
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

    if os.path.exists('out/error_digits.txt'):
        os.remove('out/error_digits.txt')

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

        videos = glob.glob(subject + '/obs-videos/*[!_cropped].flv')
        video = videos[1] if (subject_id == '034' and state == 'sober') else videos[0]
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

            path_ids, segment_ids = get_path_and_segment_ids(video, dimensions, can_data_filtered['timestamp'], video_timestamp, data_freq)
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
