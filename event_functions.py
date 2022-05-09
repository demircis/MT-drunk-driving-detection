import pandas as pd
import numpy as np

from get_features import get_features

CORES = 2

road_signs_df = pd.read_csv('out/road_sign_information.csv')

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

def calculate_event_stats(subject_data, event_type):
    scenario = subject_data.name[2]
    if event_type == 'brake':
        positive_brake_events = (subject_data[subject_data['brake'] > 0]
            .groupby((subject_data['brake'] == 0).cumsum(), as_index=False)
            .filter(lambda x: (x['gas'] == 0).all())
            .groupby((subject_data['brake'] == 0).cumsum(), as_index=False))
        brake_events_stats = positive_brake_events.apply(calculate_pedal_event_stats)
        return brake_events_stats.droplevel(0)
    elif event_type == 'gas_to_brake':
        zero_gas_events = subject_data[subject_data['gas'] == 0].groupby((subject_data['gas'] > 0).cumsum(), as_index=False)
        gas_to_brake_event = zero_gas_events.apply(lambda x: pedal_transition(x, 'brake'))
        return gas_to_brake_event.droplevel(0)
    elif event_type == 'gas':
        positive_gas_events = (subject_data[subject_data['gas'] > 0]
            .groupby((subject_data['gas'] == 0).cumsum(), as_index=False)
            .filter(lambda x: (x['brake'] == 0).all())
            .groupby((subject_data['gas'] == 0).cumsum(), as_index=False))
        gas_events_stats = positive_gas_events.apply(calculate_pedal_event_stats)
        return gas_events_stats.droplevel(0)
    elif event_type == 'brake_to_gas':
        zero_brake_events = subject_data[subject_data['brake'] == 0].groupby((subject_data['brake'] > 0).cumsum(), as_index=False)
        brake_to_gas_event = zero_brake_events.apply(lambda x: pedal_transition(x, 'gas'))
        return brake_to_gas_event.droplevel(0)
    elif event_type == 'overtaking':
        lane_zero_to_one = subject_data[subject_data['lane_number'] == 1].groupby((subject_data['lane_number'] == 0).cumsum(), as_index=False)
        overtaking_events_stats = get_overtaking_events(subject_data, lane_zero_to_one)
        if scenario == 'highway':
            lane_one_to_two = subject_data[subject_data['lane_number'] == 2].groupby((subject_data['lane_number'] == 1).cumsum(), as_index=False)
            overtaking_events_stats = pd.concat((overtaking_events_stats, get_overtaking_events(subject_data, lane_one_to_two)))
        return overtaking_events_stats.droplevel(0)
    elif event_type == 'road_sign':
        signs_for_scenario = road_signs_df[road_signs_df['scenario'] == scenario]
        speed_limit_30 = signs_for_scenario[signs_for_scenario['signType'] == SPEED_LIMIT_30]
        speed_limit_50 = signs_for_scenario[signs_for_scenario['signType'] == SPEED_LIMIT_50]
        speed_limit_60 = signs_for_scenario[signs_for_scenario['signType'] == SPEED_LIMIT_60]
        speed_limit_80 = signs_for_scenario[signs_for_scenario['signType'] == SPEED_LIMIT_80]
        speed_limit_100 = signs_for_scenario[signs_for_scenario['signType'] == SPEED_LIMIT_100]
        speed_limit_120 = signs_for_scenario[signs_for_scenario['signType'] == SPEED_LIMIT_120]

        speed_limit_30_events_stats = get_road_sign_events(speed_limit_30, subject_data, SPEED_LIMIT_30)
        speed_limit_50_events_stats = get_road_sign_events(speed_limit_50, subject_data, SPEED_LIMIT_50)
        speed_limit_60_events_stats = get_road_sign_events(speed_limit_60, subject_data, SPEED_LIMIT_60)
        speed_limit_80_events_stats = get_road_sign_events(speed_limit_80, subject_data, SPEED_LIMIT_80)
        speed_limit_100_events_stats = get_road_sign_events(speed_limit_100, subject_data, SPEED_LIMIT_100)
        speed_limit_120_events_stats = get_road_sign_events(speed_limit_120, subject_data, SPEED_LIMIT_120)

        right_of_way = signs_for_scenario[(
            (signs_for_scenario['signType'] == RIGHT_OF_WAY)
            | (signs_for_scenario['signType'] == RIGHT_OF_WAY_LEFT)
            | (signs_for_scenario['signType'] == RIGHT_OF_WAY_RIGHT))]
        stop_signs = signs_for_scenario[signs_for_scenario['signType'] == STOP_SIGN]
        speed_bumps = signs_for_scenario[signs_for_scenario['signType'] == SPEED_BUMP]
        ped_crossing_warnings = signs_for_scenario[signs_for_scenario['signType'] == PED_CROSSING_WARNING]
        ped_crossings = signs_for_scenario[signs_for_scenario['signType'] == PED_CROSSING]

        right_of_way_events_stats = get_road_sign_events(right_of_way, subject_data, RIGHT_OF_WAY)
        stop_sign_events_stats = get_road_sign_events(stop_signs, subject_data, STOP_SIGN)
        speed_bumps_events_stats = get_road_sign_events(speed_bumps, subject_data, SPEED_BUMP)
        ped_crossing_warning_events_stats = get_road_sign_events(ped_crossing_warnings, subject_data, PED_CROSSING_WARNING)
        ped_crossings_events_stats = get_road_sign_events(ped_crossings, subject_data, PED_CROSSING)

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
        return road_sign_events_stats
    elif event_type == 'turning':
        if scenario != 'highway':
            turning_events_stats = get_turning_events(subject_data)
            return turning_events_stats
        else:
            return pd.DataFrame(dtype=np.float64)


def calculate_pedal_event_stats(event):
    first = event.index.to_numpy()[0]
    last = event.index.to_numpy()[-1]
    duration = int((event['timestamp'].at[last] - event['timestamp'].at[first]).total_seconds() * 1000)
    result =  get_features(event, duration, num_cores=CORES, step_size=str(duration+1) + 'ms' if duration != 0 else '1S')
    return result.dropna(axis=0, how='all')


def pedal_transition(event, pedal):
    positive_input = event[event[pedal] > 0][pedal]
    duration = 0
    if not positive_input.empty:
        ind = positive_input.index.to_numpy()[0]
        first = event['timestamp'].index.to_numpy()[0]
        duration = int((event['timestamp'].at[ind] - event['timestamp'].at[first]).total_seconds() * 1000)
        if duration != 0:
            result = get_features(event.loc[first:ind], duration, num_cores=CORES, step_size=str(duration+1) + 'ms')
            return result.dropna(axis=0, how='all')
    result = get_features(event, duration, num_cores=CORES)
    return result.dropna(axis=0, how='all')


def get_overtaking_events(data, groupby):
    def overtaking_event(group):
        # 5 seconds before and after lane switch
        duration = 0
        if not (group.head(1).empty or group.tail(1).empty):
            start_idx = group.index.to_numpy()[0] - 150
            end_idx = group.index.to_numpy()[-1] + 150
            start, end = validate_start_end_indices(data, start_idx, end_idx)
            duration = int((data.loc[end]['timestamp'] - data.loc[start]['timestamp']).total_seconds() * 1000)
        if duration != 0:
            return get_features(data.loc[start:end], duration, num_cores=CORES, step_size=str(duration+1) + 'ms')
        else:
            return pd.DataFrame(dtype=np.float64)
    result = groupby.apply(overtaking_event)
    return result.dropna(axis=0, how='all')


def merge_maneuvers(groups, idx):
    if len(groups) == 1:
        return groups
    if idx == len(groups)-1:
        return groups
    if groups[idx+1][0] - groups[idx][-1] <= 90:
        new_groups = groups[:idx] + [groups[idx] + groups[idx+1]] + groups[idx+2:]
        return merge_maneuvers(new_groups, idx)
    else:
        return merge_maneuvers(groups, idx+1)


def get_turning_events(data):
    turning = data[(data['steer'] <= -30) | (data['steer'] >= 30)].index.values
    groups = [[turning[0]]]
    for x in turning[1:]:
        if x == groups[-1][-1] + 1:
            groups[-1].append(x)
        else:
            groups.append([x])
    
    maneuvers = merge_maneuvers(groups, 0)
    turning_event_data = []
    for maneuver_indices in maneuvers:
        start = maneuver_indices[0]
        end = maneuver_indices[-1]
        duration = int((data.loc[end]['timestamp'] - data.loc[start]['timestamp']).total_seconds() * 1000)
        if duration != 0:
            turning_event_data.append(get_features(data.loc[start:end], duration, num_cores=CORES, step_size=str(duration+1) + 'ms'))
        else:
            turning_event_data.append(get_features(data.loc[start:end], duration, num_cores=CORES))
    if len(turning_event_data) == 0:
        return pd.DataFrame(dtype=np.float64)
    result = pd.concat(turning_event_data, axis=0)
    return result.dropna(axis=0, how='all')


def get_road_sign_events(sign_info, data, sign_type):
    distances = [[(xpos - x)**2 + (ypos - y)**2 for xpos, ypos in zip(data['xpos'], data['ypos'])]
                    for x, y in zip(sign_info['sign_xPos'], sign_info['sign_yPos'])]
    distances = np.array(distances)
    if len(distances) == 0:
        return pd.DataFrame()
    indices_for_sign = np.argmin(distances, axis=1)
    road_sign_event_stats = []
    for i, min_idx in enumerate(indices_for_sign):
        if distances[i][min_idx] <= 50:
            idx = data.index.to_numpy()[min_idx]
            start_idx = idx-150
            end_idx = idx+150
            start, end = validate_start_end_indices(data, start_idx, end_idx)
            duration = int((data.loc[end]['timestamp'] - data.loc[start]['timestamp']).total_seconds() * 1000)
            if duration != 0:
                road_sign_event_stats.append(get_features(data.loc[start:end], duration, num_cores=CORES, step_size=str(duration+1) + 'ms'))
            else:
                road_sign_event_stats.append(get_features(data.loc[start:end], duration, num_cores=CORES))
    if len(road_sign_event_stats) == 0:
        return pd.DataFrame(dtype=np.float64)
    result = pd.concat(road_sign_event_stats, axis=0)
    result.dropna(axis=0, how='all', inplace=True)
    result.insert(1, 'sign_type', sign_type)
    return result


def adjust_index(df, level, subject_id, subject_state, subject_scenario):
    if not df.empty:
        df.reset_index(level=level, inplace=True)
        df.insert(0, 'subject_id', subject_id)
        df.insert(1, 'subject_state', subject_state)
        df.insert(2, 'subject_scenario', subject_scenario)
        df.set_index(['subject_id', 'subject_state', 'subject_scenario', 'datetime'], drop=True, inplace=True)
    return df


def validate_start_end_indices(data, start_idx, end_idx):
    start = start_idx
    end = end_idx
    if start_idx not in data.index:
        if data.index[data.index > start_idx].empty:
            start = data.index.max()
        else:
            start = data.index[data.index > start_idx].min()
    if end_idx not in data.index:
        if data.index[data.index < end_idx].empty:
            end = data.index.min()
        else:
            end = data.index[data.index < end_idx].max()
    return start, end
