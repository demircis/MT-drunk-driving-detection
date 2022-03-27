import pandas as pd
import numpy as np
from datetime import timedelta
from scipy.signal import find_peaks
from scipy.stats import skew, kurtosis, iqr
import multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm

def get_features(data, epoch_width=60, num_cores=0, step_size="1S"):
    if not num_cores >= 1:
        num_cores = multiprocessing.cpu_count()
    #print('using # cores: ', num_cores)

    input_data = data.copy()
    input_data.set_index('timestamp', inplace=True)

    start_time = input_data.index.min()
    end_time = input_data.index.max()

    inputs = pd.date_range(start_time, end_time, freq=step_size)

    results = Parallel(n_jobs=num_cores, backend='multiprocessing')(
        delayed(get_sliding_window)(input_data, epoch_width=epoch_width, i=k) for k in tqdm(inputs))
    results = pd.DataFrame(list(filter(None, results)))  # filter out None values
    if epoch_width == 0:
        results.insert(results.columns.get_loc('datetime')+1, 'duration', np.nan)
    else:
        results.insert(results.columns.get_loc('datetime')+1, 'duration', epoch_width / 1000.0)
    results.set_index('datetime', inplace=True)
    results.sort_index(inplace=True)

    return results


def get_sliding_window(data, epoch_width, i):

    min_timestamp = i
    max_timestamp = min_timestamp + timedelta(milliseconds=epoch_width)
    results = {
        'datetime': min_timestamp,
    }

    relevant_data = data.loc[(data.index >= min_timestamp) & (data.index < min_timestamp)]
    if epoch_width != 0:
        relevant_data = data.loc[(data.index >= min_timestamp) & (data.index < max_timestamp)]

    for column in relevant_data.columns:
        column_results = get_stats(relevant_data[column], column)
        results.update(column_results)

    return results


def get_stats(data, key_prefix=None):
    results = {
        'mean': np.nan,
        'std': np.nan,
        'min': np.nan,
        'max': np.nan,
        'q5': np.nan,
        'q95': np.nan,
        'range': np.nan,
        'iqrange': np.nan,
        'iqrange_5_95': np.nan,
        'sum': np.nan,
        'energy': np.nan,
        'skewness': np.nan,
        'kurtosis': np.nan,
        'peaks': np.nan,
        'rms': np.nan,
        'lineintegral': np.nan,
        'n_above_mean': np.nan,
        'n_below_mean': np.nan,
        'n_sign_changes': np.nan,
        'ptp': np.nan
    }

    if len(data) > 0:

        results['mean'] = np.mean(data)
        results['std'] = np.std(data)
        results['min'] = np.min(data)
        results['max'] = np.max(data)
        results['q5'] = np.quantile(data, 0.05)
        results['q95'] = np.quantile(data, 0.95)
        results['range'] = results['max'] - results['min']
        results['iqrange'] = iqr(data)
        results['iqrange_5_95'] = iqr(data, rng=(5, 95))
        results['sum'] = np.sum(data)
        results['energy'] = np.sum(data ** 2)
        results['skewness'] = skew(data)
        results['kurtosis'] = kurtosis(data)
        results['peaks'] = len(find_peaks(data, prominence=0.9)[0])
        results['rms'] = np.sqrt(results['energy'] / len(data))
        results['lineintegral'] = np.abs(np.diff(data)).sum()
        results['n_above_mean'] = np.sum(data > np.mean(data))
        results['n_below_mean'] = np.sum(data < np.mean(data))
        results['n_sign_changes'] = np.sum(np.diff(np.sign(data)) != 0)
        results['ptp'] = np.ptp(data)

    if key_prefix is not None:
        results = {key_prefix + '_' + k: v for k, v in results.items()}

    return results
