import pandas as pd
import numpy as np

from get_features import get_features

CORES = 2

def calculate_event_stats(events):
    def duration(x):
        first = x.index.to_numpy()[0]
        last = x.index.to_numpy()[-1]
        return int((x['timestamp'].at[last] - x['timestamp'].at[first]).total_seconds() * 1000)

    return events.apply(lambda x: get_features(x, duration(x), num_cores=CORES, step_size=str(duration(x)+1) + 'ms' if duration(x) != 0 else '1S'))


def brake_to_gas(x):
    gas = x[x['gas'] > 0]['gas']
    duration = 0
    if not gas.empty:
        ind = gas.index.to_numpy()[0]
        first = x['timestamp'].index.to_numpy()[0]
        duration = int((x['timestamp'].at[ind] - x['timestamp'].at[first]).total_seconds() * 1000)
        if duration != 0:
            return get_features(x.loc[first:ind], duration, num_cores=CORES, step_size=str(duration+1) + 'ms')
    return get_features(x, duration, num_cores=CORES)


def gas_to_brake(x):
    brake = x[x['brake'] > 0]['brake']
    duration = 0
    if not brake.empty:
        ind = brake.index.to_numpy()[0]
        first = x['timestamp'].index.to_numpy()[0]
        duration = int((x['timestamp'].at[ind] - x['timestamp'].at[first]).total_seconds() * 1000)
        if duration != 0:
            return get_features(x.loc[first:ind], duration, num_cores=CORES, step_size=str(duration+1) + 'ms')
    return get_features(x, duration, num_cores=CORES)


def distance_covered(x):
    if not x.empty:
        first = x['timestamp'].index.to_numpy()[0]
        last = x['timestamp'].index.to_numpy()[-1]
        return pd.Series({'distance_covered': x['velocity'].mean() * (x['timestamp'].at[last] - x['timestamp'].at[first]).total_seconds()})
    else:
        return pd.Series({'distance_covered': np.nan})


def get_overtaking_events(data, groupby):
    def overtaking_event(group):
        # 5 seconds before and after lane switch
        duration = 0
        if not (group.head(1).empty or group.tail(1).empty):
            start = max(group.index.to_numpy()[0] - 150, 0)
            end = min(group.index.to_numpy()[-1] + 150, data.shape[0]-1)
            duration = int((data.iloc[end]['timestamp'] - data.iloc[start]['timestamp']).total_seconds() * 1000)
        if duration != 0:
            return get_features(data.iloc[start:end], duration, num_cores=CORES, step_size=str(duration+1) + 'ms')
        else:
            return get_features(data.iloc[start:end], duration, num_cores=CORES)
    return groupby.apply(overtaking_event)


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


def get_turning_events(data, subject_id, subject_state, subject_scenario):
    turning = data[(data['steer'] <= -15) | (data['steer'] >= 15)].index.values
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
        duration = int((data.loc[end, 'timestamp'] - data.loc[start, 'timestamp']).total_seconds() * 1000)
        if duration != 0:
            turning_event_data.append(get_features(data.loc[start:end], duration, num_cores=CORES, step_size=str(duration+1) + 'ms'))
        else:
            turning_event_data.append(get_features(data.loc[start:end], duration, num_cores=CORES))
    if len(turning_event_data) == 0:
        return pd.DataFrame()
    df = pd.concat(turning_event_data, axis=0)
    df.dropna(axis=0, how='all', inplace=True)
    df = adjust_index(df, 0, subject_id, subject_state, subject_scenario)
    return df


def get_road_sign_events(sign_info, data, sign_type, subject_id, subject_state, subject_scenario):
    distances = [[(xpos - x)**2 + (ypos - y)**2 for xpos, ypos in zip(data['xpos'], data['ypos'])]
                    for x, y in zip(sign_info['sign_xPos'], sign_info['sign_yPos'])]
    distances = np.array(distances)
    if len(distances) == 0:
        return pd.DataFrame()
    indices_for_sign = np.argmin(distances, axis=1)
    road_sign_event_stats = []
    for i, idx in enumerate(indices_for_sign):
        if distances[i][idx] <= 50:
            start = max(0, idx-150)
            end = min(idx+150, data.shape[0]-1)
            duration = int((data.iloc[end]['timestamp'] - data.iloc[start]['timestamp']).total_seconds() * 1000)
            if duration != 0:
                road_sign_event_stats.append(get_features(data.iloc[start:end], duration, num_cores=CORES, step_size=str(duration+1) + 'ms'))
            else:
                road_sign_event_stats.append(get_features(data.iloc[start:end], duration, num_cores=CORES))
    if len(road_sign_event_stats) == 0:
        return pd.DataFrame()
    df = pd.concat(road_sign_event_stats, axis=0)
    df.dropna(axis=0, how='all', inplace=True)
    df = adjust_index(df, 0, subject_id, subject_state, subject_scenario)
    df.insert(1, 'sign_type', sign_type)
    return df


def adjust_index(df, level, subject_id, subject_state, subject_scenario):
    if not df.empty:
        df.reset_index(level=level, inplace=True)
        df.insert(0, 'subject_id', subject_id)
        df.insert(1, 'subject_state', subject_state)
        df.insert(2, 'subject_scenario', subject_scenario)
        df.set_index(['subject_id', 'subject_state', 'subject_scenario', 'datetime'], drop=True, inplace=True)
    return df
