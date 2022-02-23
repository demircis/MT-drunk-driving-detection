import pandas as pd
import numpy as np
from scipy.stats import kurtosis
from scipy.signal import find_peaks, peak_widths

def calculate_event_stats(events, signal):
    def start(x):
        return x.head(1)
    def end(x):
        return x.tail(1)
    def duration(x):
        return (x.max()-x.min()).total_seconds()
    def quartile(x):
        return np.quantile(x, q=0.25) if not x.empty else np.nan
    def autocorr(x):
        return x.autocorr() if not x.empty else np.nan
    def mean_peak_heights(x):
        _, props = find_peaks(x, height=0, width=1, plateau_size=0)
        return np.mean(props['peak_heights']) if not len(props['peak_heights']) == 0 else np.nan
    def mean_peak_widths(x):
        ind, _ = find_peaks(x, height=0, width=1, plateau_size=0)
        widths, _, _, _ = peak_widths(x, ind)
        return np.mean(widths) if not len(widths) == 0 else np.nan
    
    return events.agg(
            {
                'timestamp': [start, duration],
                signal: ['mean', 'min', 'max', 'std', 'skew', kurtosis, quartile, autocorr, mean_peak_heights, mean_peak_widths],
                'velocity': [start, 'mean', 'std', end],
                'acc': [start, 'mean', 'std', end],
                'steer': [start, 'mean', 'std', end],
                'SteerSpeed': [start, 'mean', 'std', end],
                'latvel': [start, 'mean', 'std', end]
            }
        )


def brake_to_gas(x):
    gas = x[x['gas'] > 0]['gas']
    if not gas.empty:
        ind = gas.index.to_numpy()[0]
        first = x['timestamp'].index.to_numpy()[0]
        return pd.Series({'brake_to_gas': (x['timestamp'].at[ind] - x['timestamp'].at[first]).total_seconds()})
    else:
        return pd.Series({'brake_to_gas': np.nan})


def gas_to_brake(x):
    brake = x[x['brake'] > 0]['brake']
    if not brake.empty:
        ind = brake.index.to_numpy()[0]
        first = x['timestamp'].index.to_numpy()[0]
        return pd.Series({'gas_to_brake': (x['timestamp'].at[ind] - x['timestamp'].at[first]).total_seconds()})
    else:
        return pd.Series({'gas_to_brake': np.nan})


def distance_covered(x):
    if not x.empty:
        first = x['timestamp'].index.to_numpy()[0]
        last = x['timestamp'].index.to_numpy()[-1]
        return pd.Series({'distance_covered': x['velocity'].mean() * (x['timestamp'].at[last] - x['timestamp'].at[first]).total_seconds()})
    else:
        return pd.Series({'distance_covered': np.nan})


def get_overtaking_event(data, groupby):
    def overtaking_event(group):
        # 5 seconds before and after lane switch
        if not (group.head(1).empty or group.tail(1).empty):
            start = max(group.head(1).index.to_numpy()[0] - 150, 0)
            end = min(group.tail(1).index.to_numpy()[0] + 150, data.index.max())
            timestamp = data.iloc[start]['timestamp']
            duration = (data.iloc[end]['timestamp'] - data.iloc[start]['timestamp']).total_seconds()
            distance = data.iloc[start:end]['velocity'].mean() * duration
            mean_lane_pos = data.iloc[start:end]['lane_position'].mean()
            std_lane_pos = data.iloc[start:end]['lane_position'].std()
            min_velocity = data.iloc[start:end]['velocity'].min()
            max_velocity = data.iloc[start:end]['velocity'].max()
            mean_velocity = data.iloc[start:end]['velocity'].mean()
            std_velocity = data.iloc[start:end]['velocity'].std()
            min_acc = data.iloc[start:end]['acc'].min()
            max_acc = data.iloc[start:end]['acc'].max()
            mean_acc = data.iloc[start:end]['acc'].mean()
            std_acc = data.iloc[start:end]['acc'].std()
            min_steer = data.iloc[start:end]['steer'].min()
            max_steer = data.iloc[start:end]['steer'].max()
            mean_steer = data.iloc[start:end]['steer'].mean()
            std_steer = data.iloc[start:end]['steer'].std()
            min_SteerSpeed = data.iloc[start:end]['SteerSpeed'].min()
            max_SteerSpeed = data.iloc[start:end]['SteerSpeed'].max()
            mean_SteerSpeed = data.iloc[start:end]['SteerSpeed'].mean()
            std_SteerSpeed = data.iloc[start:end]['SteerSpeed'].std()
            min_latvel = data.iloc[start:end]['latvel'].min()
            max_latvel = data.iloc[start:end]['latvel'].max()
            mean_latvel = data.iloc[start:end]['latvel'].mean()
            std_latvel = data.iloc[start:end]['latvel'].std()
            return pd.Series({
                'timestamp': timestamp,
                'duration': duration,
                'distance': distance,
                'mean_lane_position': mean_lane_pos,
                'std_lane_position': std_lane_pos,
                'min_velocity': min_velocity,
                'max_velocity': max_velocity,
                'mean_velocity': mean_velocity,
                'std_velocity': std_velocity,
                'min_acc': min_acc,
                'max_acc': max_acc,
                'mean_acc': mean_acc,
                'std_acc': std_acc,
                'min_steer': min_steer,
                'max_steer': max_steer,
                'mean_steer': mean_steer,
                'std_steer': std_steer,
                'min_SteerSpeed': min_SteerSpeed,
                'max_SteerSpeed': max_SteerSpeed,
                'mean_SteerSpeed': mean_SteerSpeed,
                'std_SteerSpeed': std_SteerSpeed,
                'min_latvel': min_latvel,
                'max_latvel': max_latvel,
                'mean_latvel': mean_latvel,
                'std_latvel': std_latvel
                })
        else:
            return pd.Series({
                'timestamp': np.nan,
                'duration': np.nan,
                'distance': distance,
                'mean_lane_position': mean_lane_pos,
                'std_lane_position': std_lane_pos,
                'min_velocity': np.nan,
                'max_velocity': np.nan,
                'mean_velocity': np.nan,
                'std_velocity': np.nan,
                'min_acc': np.nan,
                'max_acc': np.nan,
                'mean_acc': np.nan,
                'std_acc': np.nan,
                'min_steer': np.nan,
                'max_steer': np.nan,
                'mean_steer': np.nan,
                'std_steer': np.nan,
                'min_SteerSpeed': np.nan,
                'max_SteerSpeed': np.nan,
                'mean_SteerSpeed': np.nan,
                'std_SteerSpeed': np.nan,
                'min_latvel': np.nan,
                'max_latvel': np.nan,
                'mean_latvel': np.nan,
                'std_latvel': np.nan
                })
    return groupby.apply(overtaking_event)