import pandas as pd
import numpy as np
from sklearnex import patch_sklearn
patch_sklearn()
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from xgboost import XGBRFClassifier
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import cross_validate
from sklearn.model_selection import LeaveOneGroupOut

RANDOM_STATE = 42

SCORING = ['balanced_accuracy', 'roc_auc', 'f1_micro', 'average_precision', 'recall', 'precision']

SIGNAL_COMBOS = [['driver_behavior', 'vehicle_behavior'], ['driver_behavior', 'vehicle_behavior', 'navi'],
                ['driver_behavior', 'vehicle_behavior', 'radar'], ['driver_behavior', 'vehicle_behavior', 'navi', 'radar']]

EVENTS = ['brake', 'brake_to_gas', 'gas', 'gas_to_brake', 'overtaking', 'road_sign', 'turning']

SELECTED_SIGNALS = [
    'brake',
    'brake_acc',
    'brake_jerk',
    'brake_vel',
    'gas',
    'gas_acc',
    'gas_jerk',
    'gas_vel',
    'steer',
    'SteerSpeed',
    'SteerSpeed_acc',
    'SteerSpeed_jerk',
    'speed_limit_exceeded',
    'SpeedDif',
    'Dhw',
    'is_crossing_lane_left',
    'is_crossing_lane_right',
    'lane_crossing',
    'lane_distance_left_edge',
    'lane_distance_right_edge',
    'Ttc',
    'TtcOpp',
    'acc',
    'acc_jerk',
    'velocity',
    'latvel_acc',
    'latvel_jerk',
    'YawRate_acc',
    'YawRate_jerk',
    'YawRate'
]

STATS = ['mean', 'std', 'min', 'max', 'q5', 'q95', 'iqrange', 'iqrange_5_95', 'skewness', 'kurtosis', 'peaks', 'rms']

SUM_COLUMNS = ['lane_crossing', 'lane_crossing_left', 'lane_crossing_right', 'is_crossing_lane', 'is_crossing_lane_left', 'is_crossing_lane_right', 'speed_limit_exceeded']

SCENARIOS = ['highway', 'rural', 'town']

LOGO = LeaveOneGroupOut()

def do_sliding_window_classification(window_sizes, overlap_percentages, classifier, mode):
    for window_size in window_sizes:
        for combo in SIGNAL_COMBOS:
            signal_string = ''
            can_data_features = []
            for signal in combo:
                signal_string += '_' + signal
                can_data_features.append(pd.read_parquet('out/can_data_features_{}_windowsize_{}s.parquet'.format(signal, window_size)))
            can_data_features = pd.concat(can_data_features, axis=1)
            
            if classifier == 'log_regression':
                can_data_features.dropna(axis=1, inplace=True)

            for overlap_percentage in overlap_percentages:
                step = 1
                if overlap_percentage is not None:
                    step = window_size - int(overlap_percentage * window_size)
                can_data_features_step = can_data_features[(can_data_features.groupby(['subject_id', 'subject_state', 'subject_scenario']).cumcount() % step) == 0]

                for scenario in SCENARIOS:
                    print('signals: {}, window size: {}s, step size: {}s ({}), scenario: {}'.format(
                        signal_string, window_size, step, overlap_percentage, scenario
                        ))
                    
                    can_data_features_step = can_data_features_step[select_columns(can_data_features_step)]

                    can_data_features_scenario = can_data_features_step.loc[:, :, scenario, :]

                    X, y, weights, groups = prepare_dataset(can_data_features_scenario, mode)

                    clf = get_classifier(classifier, mode)
                    
                    subject_ids = np.unique(groups)

                    max_features = 50
                    best_X, selected_features = sequential_feature_selection(max_features, clf, can_data_features_scenario, X, y, groups, weights, len(subject_ids)-1)
                    selected_features.to_csv(
                            'out/results/{}_{}_selected_features_windowsize_{}_step_size_{}s{}_{}.csv'.format(
                                classifier, mode, window_size, step, signal_string, scenario
                                ), index=True, header=['selected_features']
                            )

                    cv = cross_validate(estimator=clf, X=best_X, y=y, scoring=SCORING, return_estimator=True, verbose=0,
                            return_train_score=True, cv=LOGO, groups=groups, n_jobs=len(subject_ids)-1, fit_params={'sample_weight': weights})
                    
                    results = collect_results(cv, subject_ids)
                    results.to_csv(
                            'out/results/{}_{}_pred_results_windowsize_{}_step_size_{}s{}_{}.csv'.format(
                                classifier, mode, window_size, step, signal_string, scenario
                                ), index=True, header=True
                            )


def do_event_classification(classifier, mode):
    can_data_events = None
    for event in EVENTS:
        can_data_events = pd.read_parquet('out/can_data_{}_events.parquet'.format(event))
        #can_data_events = pd.concat(can_data_events, axis=0)

        if classifier == 'log_regression':
            can_data_events.dropna(axis=1, inplace=True)

        for scenario in SCENARIOS:
            print('event type: {}, scenario: {}'.format(event, scenario))

            if event == 'turning' and scenario == 'highway':
                continue

            can_data_events = can_data_events[['duration'] + select_columns(can_data_events)]

            can_data_events_scenario = can_data_events.loc[:, :, scenario, :]

            X, y, weights, groups = prepare_dataset(can_data_events_scenario, mode)

            clf = get_classifier(classifier, mode)
            
            subject_ids = np.unique(groups)

            max_features = 20
            best_X, selected_features = sequential_feature_selection(max_features, clf, can_data_events_scenario, X, y, groups, weights, len(subject_ids)-1)
            selected_features.to_csv(
                    'out/results/{}_{}_selected_features_{}_{}.csv'.format(
                        classifier, mode, event, scenario
                        ), index=True, header=['selected_features']
                    )

            cv = cross_validate(estimator=clf, X=best_X, y=y, scoring=SCORING, return_estimator=True, verbose=0,
                    return_train_score=True, cv=LOGO, groups=groups, n_jobs=len(subject_ids)-1, fit_params={'sample_weight': weights})

            results = collect_results(cv, subject_ids)
            results.to_csv(
                    'out/results/{}_{}_pred_results_{}_{}.csv'.format(
                        classifier, mode, event, scenario), index=True, header=True
                    )


def do_combined_classification(classifier, window_sizes, mode):
    for window_size in window_sizes:
        for combo in SIGNAL_COMBOS:
            signal_string = ''
            can_data_features = []
            for signal in combo:
                signal_string += '_' + signal
                can_data_features.append(pd.read_parquet('out/can_data_features_{}_windowsize_{}s.parquet'.format(signal, window_size)))
            can_data_features = pd.concat(can_data_features, axis=1)

            can_data_features = can_data_features.loc[:,~can_data_features.columns.duplicated()]
            can_data_features = can_data_features[['duration'] + select_columns(can_data_features)]
            
            can_data_events = []
            for event in EVENTS:
                can_data_events.append(pd.read_parquet('out/can_data_{}_events.parquet'.format(event)))
            can_data_events = pd.concat(can_data_events, axis=0)

            can_data_events = can_data_events[['duration'] + select_columns(can_data_events)]

            can_data_combined = pd.concat((can_data_features, can_data_events), axis=0)

            if classifier == 'log_regression':
                can_data_combined.dropna(axis=1, inplace=True)

            for scenario in SCENARIOS:
                print('signals: {}, window size: {}s, scenario: {}'.format(
                    signal_string, window_size, scenario
                    ))
                
                can_data_combined_scenario = can_data_combined.loc[:, :, scenario, :]

                X, y, weights, groups = prepare_dataset(can_data_combined_scenario, mode)

                clf = get_classifier(classifier, mode)

                subject_ids = np.unique(groups)

                max_features = 50
                best_X, selected_features = sequential_feature_selection(max_features, clf, can_data_combined_scenario, X, y, groups, weights, len(subject_ids)-1)
                selected_features.to_csv(
                        'out/results/{}_{}_selected_features_combined_windowsize_{}{}_{}.csv'.format(
                            classifier, mode, window_size, signal_string, scenario
                            ), index=True, header=['selected_features']
                        )

                cv = cross_validate(estimator=clf, X=best_X, y=y, scoring=SCORING, return_estimator=True, verbose=0,
                        return_train_score=True, cv=LOGO, groups=groups, n_jobs=len(subject_ids), fit_params={'sample_weight': weights})

                results = collect_results(cv, subject_ids)
                results.to_csv(
                            'out/results/{}_{}_pred_results_combined_windowsize_{}{}_{}.csv'.format(
                                classifier, mode, window_size, signal_string, scenario
                                ), index=True, header=True
                            )


def select_columns(data):
    stat_columns_list = [
        [col for col in data.columns if col in 
            [sig + '_' + s for s in (STATS + ['sum'] if sig in SUM_COLUMNS else STATS)]
        ] for sig in SELECTED_SIGNALS
    ]
    stat_columns = []
    for item in stat_columns_list:
        stat_columns += item
    return stat_columns


def prepare_dataset(data, mode):
    input_data = data.copy()
    input_data.loc[:, 'label'] = 0
    if mode == 'binary':
        input_data.loc[(slice(None), 'above', slice(None), slice(None)), 'label'] = 1
        # drop below BAC level for binary classification
        input_data.drop('below', level=1, inplace=True)
    elif mode == 'multiclass':
        input_data.loc[(slice(None), 'below', slice(None), slice(None)), 'label'] = 1
        input_data.loc[(slice(None), 'above', slice(None), slice(None)), 'label'] = 2
    else:
        raise ValueError('Received unknown classifier mode string!')

    X = input_data.drop(columns='label').to_numpy(dtype=np.float64)
    
    y = input_data['label'].to_numpy()
    class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
    weights = np.zeros(y.shape)
    for i, weight in enumerate(class_weights):
        weights[y == i] = weight

    groups = list(input_data.index.get_level_values('subject_id'))

    X = StandardScaler().fit_transform(X, y)

    return X, y, weights, groups


def sequential_feature_selection(max_features, estimator, data, X, y, groups=None, weights=None, n_jobs=1):
    sfs = SequentialFeatureSelector(estimator, k_features=(1, max_features), scoring='balanced_accuracy', cv=LOGO, n_jobs=n_jobs, verbose=2)
    best_X = sfs.fit_transform(X, y, groups=groups, sample_weight=weights)
    print('\nbest score (with {} features): {}'.format(len(list(sfs.k_feature_idx_)), sfs.k_score_))
    selected_features = pd.Series(data.columns[list(sfs.k_feature_idx_)])
    print(selected_features.to_numpy())
    return best_X, selected_features


def get_classifier(classifier, mode):
    clf = None
    if mode == 'binary':
        if classifier == 'log_regression':
            clf = LogisticRegression(
            penalty='l1', solver='saga', max_iter=1000, tol=1e-2, random_state=RANDOM_STATE)
        elif classifier == 'random_forest':
            clf = XGBRFClassifier(objective='binary:logistic', eval_metric='logloss', n_estimators=100, use_label_encoder=False, n_jobs=1, random_state=RANDOM_STATE)
        else:
            raise ValueError('Received unknown classifier string!')
    elif mode == 'multiclass':
        if classifier == 'log_regression':
            clf = LogisticRegression(
            penalty='l1', solver='saga', max_iter=1000, tol=1e-2, random_state=RANDOM_STATE)
        elif classifier == 'random_forest':
            clf = XGBRFClassifier(objective='multi:softmax', n_estimators=100, use_label_encoder=False, n_jobs=1, random_state=RANDOM_STATE)
        else:
            raise ValueError('Received unknown classifier string!')
    return clf


def collect_results(cv_object, subject_ids):
    results = pd.DataFrame({k:v for k,v in cv_object.items() if k not in ['estimator']}).set_index(subject_ids)
    mean = results.mean(axis=0).rename('mean')
    std = results.std(axis=0).rename('stddev')
    results = results.append(mean)
    results = results.append(std)
    return results
