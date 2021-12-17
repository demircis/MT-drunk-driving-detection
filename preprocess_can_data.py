import pandas as pd
import glob
import re
import datetime

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

    return can_data_filtered

def split_signal_in_right_left(df, signal):
    df[signal + "_right"] = df.loc[df[signal] < 0, signal].abs()
    df[signal + "_right"] = df[signal + "_right"].fillna(0)
    df[signal + "_left"] = df.loc[df[signal] > 0, signal]
    df[signal + "_left"] = df[signal + "_right"].fillna(0)

    return df


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
        df[signal_new + suffix] = df[signal].diff() / time_delta
        df[signal_new + suffix] = df[signal_new + suffix].fillna(0)

    return df


def do_preprocessing(full_study, data_freq=30):
    if glob.glob('out/can_data.parquet'):
        return

    CAN_COLUMNS = ['interval', 'steer', 'latpos', 'gas', 'brake', 'clutch', 'Thw', 'velocity', 'acc', 'latvel', 'dtoint', 'indicator',
               'heading', 'SpeedDif', 'LaneDirection', 'SteerError', 'SteerSpeed', 'Ttc', 'TtcOpp', 'LeftDis',
               'RightDis', 'AheadDis', 'traflight', 'handbrake', 'engine']

    SIGNALS_WITH_POSITIVE_AND_NEGATIVE_VALUES = ["latpos", "steer", "latvel", "SteerSpeed", "SteerError"]

    SIGNALS_SLOPES = ['gas', 'brake']
    SIGNALS_DERIVE_ACCELERATION = ['latvel', 'SteerSpeed']
    SIGNALS_DERIVE_JERK = ['latvel_acc', 'SteerSpeed_acc']

    data = []

    if full_study:
        DATA_FOLDER = "/mnt/adar/drive/study"
        CAN_COLUMNS = CAN_COLUMNS[:22] + ["ypos", "xpos", "YawRate"] + CAN_COLUMNS[22:]
        SIGNALS_WITH_POSITIVE_AND_NEGATIVE_VALUES = SIGNALS_WITH_POSITIVE_AND_NEGATIVE_VALUES + ["YawRate"]
        SIGNALS_DERIVE_ACCELERATION = SIGNALS_DERIVE_ACCELERATION + ["YawRate"]
        SIGNALS_DERIVE_JERK = SIGNALS_DERIVE_JERK + ["YawRate_acc"]

        subject_folders = sorted(glob.glob(DATA_FOLDER + '/*_sober') + glob.glob(DATA_FOLDER + '/*_above') + glob.glob(
            DATA_FOLDER + '/*_below'))
        subject_folders = [f for f in subject_folders if 'audio' not in f]
    else:
        DATA_FOLDER = "/mnt/adar/drive/pilot"
        subject_folders = sorted(glob.glob(DATA_FOLDER + '/*_drunk') + glob.glob(DATA_FOLDER + '/*_sober'))

    subject_re = re.compile('d-([0-9]+)')
    scenario_re = re.compile(r's\d{2}[abcde]-(\w*)Exp.dat$')
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
            can_data['interval'] = [pd.to_datetime(simulator_timestamp + datetime.timedelta(milliseconds=interval * 1000))
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
            can_data_filtered = can_data_filtered.drop(["indicator"], axis=1)

            can_data_filtered = do_derivation_of_signals(can_data_filtered, SIGNALS_SLOPES, '_slope', data_freq)
            can_data_filtered = do_derivation_of_signals(can_data_filtered, SIGNALS_DERIVE_ACCELERATION, '_acc', data_freq)
            can_data_filtered = do_derivation_of_signals(can_data_filtered, SIGNALS_DERIVE_JERK, '_jerk', data_freq, '_acc')

            data.append(can_data_filtered)
    
    data = pd.concat(data)
    data.to_parquet("out/can_data.parquet")
