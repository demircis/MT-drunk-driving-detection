import pandas as pd
import numpy as np
from datetime import timedelta
from scipy.signal import find_peaks
from scipy.stats import skew, kurtosis, iqr
import multiprocessing
from joblib import Parallel, delayed

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
        delayed(get_sliding_window)(input_data, epoch_width=epoch_width, i=k) for k in inputs)
    results = pd.DataFrame(list(filter(None, results)))  # filter out None values
    results.set_index('datetime', inplace=True)
    results.sort_index(inplace=True)

    return results


def get_sliding_window(data, epoch_width, i):

    min_timestamp = i
    max_timestamp = min_timestamp + timedelta(milliseconds=epoch_width * 1000)
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
        'q5': np.nan,
        'max': np.nan,
        'q95': np.nan,
        'range': np.nan,
        'iqrange': np.nan,
        'sum': np.nan,
        'energy': np.nan,
        'skewness': np.nan,
        'kurtosis': np.nan,
        'peaks': np.nan,
        'rms': np.nan,
        'lineintegral': np.nan,
    }

    if len(data) > 0:

        results['mean'] = np.mean(data)
        results['std'] = np.std(data)
        results['min'] = np.min(data)
        results['q5'] = np.quantile(data, 0.05)
        results['max'] = np.max(data)
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

    if key_prefix is not None:
        results = {key_prefix + '_' + k: v for k, v in results.items()}

    return results
