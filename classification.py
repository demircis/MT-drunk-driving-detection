import os
import pandas as pd
import numpy as np
from sklearnex import patch_sklearn
patch_sklearn()
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBRFClassifier
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import cross_validate
from sklearn.model_selection import LeaveOneGroupOut

random_state = 42

from yaml import load, Loader
from bunch import Bunch
Bunch.__str__ = Bunch.__repr__

stream = open("config.yaml", 'r')
config = Bunch(load(stream, Loader=Loader))

SCORING = ['balanced_accuracy', 'roc_auc', 'f1_micro', 'average_precision', 'recall', 'precision']

SIGNAL_COMBOS = [['driver_behavior', 'vehicle_behavior'], ['driver_behavior', 'vehicle_behavior', 'navi'],
                ['driver_behavior', 'vehicle_behavior', 'radar'], ['driver_behavior', 'vehicle_behavior', 'navi', 'radar']]

EVENTS = ['brake', 'brake_to_gas', 'gas', 'gas_to_brake', 'overtaking', 'road_sign', 'turning']

SELECTED_FEATURES = [
    'brake_jerk',
    'brake_vel',
    'gas',
    'gas_acc',
    'gas_jerk',
    'gas_vel',
    'SteerSpeed',
    'SteerSpeed_acc',
    'SteerSpeed_jerk',
    'speed_limit_exceeded',
    'SpeedDif',
    'is_crossing_lane',
    'lane_distance_left_edge',
    'lane_distance_right_edge',
    'lane_position',
    'acc_jerk',
    'latvel_jerk',
    'YawRate',
    'YawRate_acc',
    'YawRate_jerk'
]

STATS = ['mean', 'std', 'skewness', 'kurtosis', 'rms', 'q5', 'q95', 'min', 'max', 'peaks', 'range', 'iqrange', 'iqrange_5_95']

SUM_COLUMNS = ['lane_crossing', 'is_crossing_lane', 'is_crossing_lane_left', 'is_crossing_lane_right', 'speed_limit_exceeded']

SCENARIOS = ['highway', 'rural', 'town']

LOGO = LeaveOneGroupOut()


def do_sliding_window_classification(window_sizes, overlap_percentages, classifier, mode):
    for window_size in window_sizes:
        for combo in SIGNAL_COMBOS:
            signal_string = ''
            can_data_features = []
            for signal in combo:
                signal_string += '_' + signal
                can_data_features.append(pd.read_parquet('out/can_data_features_{}_windowsize_{}s_new.parquet'.format(signal, window_size)))
            can_data_features = pd.concat(can_data_features, axis=1)
            stat_columns_list = [
                [col for col in can_data_features.columns if col in 
                    [sig + '_' + s for s in (STATS + ['sum'] if sig in SUM_COLUMNS else STATS)]
                ] for sig in SELECTED_FEATURES
            ]
            stat_columns = []
            for item in stat_columns_list:
                stat_columns += item
            can_data_features = can_data_features[stat_columns]
            
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

                    can_data_features_bin = can_data_features_step.loc[:, :, scenario, :]

                    X, y, weights, groups = prepare_dataset(can_data_features_bin, mode)

                    clf = get_classifier(classifier, mode)
                    
                    subject_ids = np.unique(groups)

                    max_features = 50
                    sfs = SequentialFeatureSelector(clf, k_features=(1, max_features), scoring='balanced_accuracy', cv=LOGO, n_jobs=len(subject_ids), verbose=2)
                    best_X = sfs.fit_transform(X, y, groups=groups, sample_weight=weights)
                    print('\nbest score (with {} features): {}'.format(len(list(sfs.k_feature_idx_)), sfs.k_score_))
                    print(can_data_features_bin.columns.to_numpy()[list(sfs.k_feature_idx_)])

                    cv = cross_validate(estimator=clf, X=best_X, y=y, scoring=SCORING, return_estimator=True, verbose=0,
                            return_train_score=True, cv=LOGO, groups=groups, n_jobs=len(subject_ids), fit_params={'sample_weight': weights})

                    # for subject_id, est in zip(subject_ids, cv['estimator']):
                    #     if not os.path.exists('out/results/subject_{}'.format(subject_id)):
                    #         os.makedirs('out/results/subject_{}'.format(subject_id))
                    #     RocCurveDisplay.from_estimator(est, X, y)
                    #     plt.savefig('out/results/subject_{}/roc_curve_windowsize_{}{}_{}.png'.format(subject_id, window_size, signal_string, scenario))
                    #     plt.close()

                    # shap_values = np.zeros(can_data_features_bin.shape)
                    # for ind in range(len(subject_ids)):
                    #     feature_names = can_data_features_bin.columns.to_list()[:-1]
                    #     explainer = shap.LinearExplainer(cv['estimator'][ind]['logisticregression'], X, feature_names=feature_names)
                    #     shap_values += explainer.shap_values(X)
                    
                    # shap_values /= len(range(subject_ids))

                    # shap_values = pd.DataFrame(shap_values)
                    # shap_values.columns = feature_names

                    # ind = random.choice(range(len(subject_ids)))
                    # feature_names = can_data_features_bin.columns.to_list()[:-1]
                    # explainer = shap.Explainer(cv['estimator'][ind]['logisticregression'], X, feature_names=feature_names)
                    # shap_obj = explainer(X)
                    # shap.plots.beeswarm(shap_obj)

                    # shap_values.set_index(can_data_features_bin.index).to_parquet(
                    #     'out/shap_values_windowsize_{}{}_{}.parquet'.format(window_size, signal_string, scenario), index=True
                    # )

                    # indices = cv['estimator'][0]['selectfrommodel'].get_support(indices=True)
                    # print(can_data_features_bin.columns[indices])
                    
                    results = collect_results(cv, subject_ids)
                    results.to_csv(
                            'out/results/{}_pred_results_windowsize_{}_step_size_{}s{}_{}.csv'.format(
                                classifier, window_size, step, signal_string, scenario
                                ), index=True, header=True
                            )


def do_event_classification(classifier, mode):
    can_data_events = []
    for e in EVENTS:
        can_data_events.append(pd.read_parquet('out/can_data_{}_events.parquet'.format(e)))
    can_data_events = pd.concat(can_data_events, axis=0)
    stat_columns_list = [
                [col for col in can_data_events.columns if col in 
                    [sig + '_' + s for s in (STATS + ['sum'] if sig in SUM_COLUMNS else STATS)]
                ] for sig in SELECTED_FEATURES
            ]
    stat_columns = []
    for item in stat_columns_list:
        stat_columns += item
    can_data_events = can_data_events[['duration'] + stat_columns]

    if classifier == 'log_regression':
        can_data_events.dropna(axis=1, inplace=True)

    for k, scenario in enumerate(SCENARIOS):
        print('scenario: {}'.format(scenario))

        can_data_events_bin = can_data_events.loc[:, :, scenario, :]

        X, y, weights, groups = prepare_dataset(can_data_events_bin, mode)

        clf = get_classifier(classifier, mode)
        
        subject_ids = np.unique(groups)

        #sfs = SequentialFeatureSelector(clf, n_features_to_select=5, scoring='roc_auc', n_jobs=len(subject_ids))

        cv = cross_validate(estimator=clf, X=X, y=y, scoring=SCORING, return_estimator=True, verbose=0,
                return_train_score=True, cv=LOGO, groups=groups, n_jobs=len(subject_ids))

        results = collect_results(cv, subject_ids)
        results.to_csv(
                'out/results/{}_pred_results_events_{}.csv'.format(
                    classifier, scenario), index=True, header=True
                )


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
        weights[y == i] = class_weights[i]

    groups = list(input_data.index.get_level_values('subject_id'))

    X = StandardScaler().fit_transform(X, y)

    return X, y, weights, groups


def get_classifier(classifier, mode):
    clf = None
    if mode == 'binary':
        if classifier == 'log_regression':
            clf = LogisticRegression(
            penalty='l1', solver='saga', max_iter=1000, tol=1e-2, random_state=random_state)
        elif classifier == 'random_forest':
            clf = XGBRFClassifier(objective='binary:logistic', eval_metric='logloss', n_estimators=100, use_label_encoder=False, n_jobs=1, random_state=random_state)
        else:
            raise ValueError('Received unknown classifier string!')
    elif mode == 'multiclass':
        if classifier == 'log_regression':
            clf = LogisticRegression(
            penalty='l1', solver='saga', max_iter=1000, tol=1e-2, random_state=random_state)
        elif classifier == 'random_forest':
            clf = XGBRFClassifier(objective='multi:softmax', n_estimators=100, use_label_encoder=False, n_jobs=1, random_state=random_state)
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
