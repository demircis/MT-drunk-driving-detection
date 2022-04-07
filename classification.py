import os
import pandas as pd
import numpy as np
from sklearnex import patch_sklearn
patch_sklearn()
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import cross_validate
from sklearn.model_selection import LeaveOneGroupOut

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
    'gas',
    'brake',
    'SteerSpeed',
    'gas_vel',
    'brake_vel',
    'gas_acc',
    'gas_jerk',
    'acc',
    'acc_jerk',
    'lat_vel',
    'lane_position',
    'lane_crossing',
    'is_crossing_lane_left',
    'is_crossing_lane_right',
    'Ttc',
    'TtcOpp',
    'Thw',
    'Dhw',
    'SpeedDif',
    'speed_limit_exceeded'
]

#STATS = ['mean', 'std', 'min','max', 'q5', 'q95', 'range', 'iqrange', 'iqrange_5_95', 'sum', 'energy', 'skewness',
         #'kurtosis', 'peaks', 'rms', 'lineintegral', 'n_above_mean', 'n_below_mean', 'n_sign_changes', 'ptp']
STATS = ['mean', 'std', 'skewness', 'kurtosis', 'rms', 'q5', 'q95', 'min', 'max', 'peaks']

SUM_COLUMNS = ['lane_crossing', 'is_crossing_lane_left', 'is_crossing_lane_right', 'speed_limit_exceeded']

SCENARIOS = ['highway', 'rural', 'town']

LOGO = LeaveOneGroupOut()


def do_sliding_window_classification(window_sizes, overlap_percentages, classifier):
    for i, window_size in enumerate(window_sizes):
        for j, combo in enumerate([SIGNAL_COMBOS[-1]]):
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
            can_data_features.loc[:, 'label'] = 0
            can_data_features.loc[(slice(None), 'above', slice(None), slice(None)), 'label'] = 1

            can_data_features.dropna(axis=1, inplace=True)

            # drop below BAC level for binary classification
            can_data_features = can_data_features.drop('below', level=1)

            for overlap_percentage in overlap_percentages:
                step = 1
                if overlap_percentage is not None:
                    step = window_size - int(overlap_percentage * window_size)
                can_data_features_step = can_data_features[(can_data_features.groupby(['subject_id', 'subject_state', 'subject_scenario']).cumcount() % step) == 0]

                for k, scenario in enumerate(SCENARIOS):
                    print('signals: {}, window size: {}s, step size: {}s ({}), scenario: {}'.format(
                        signal_string, window_size, step, overlap_percentage, scenario
                        ))

                    can_data_features_bin = can_data_features_step.loc[:, :, scenario, :]

                    groups = list(can_data_features.index.get_level_values('subject_id'))
                    subject_ids = np.unique(groups)
                    
                    X = can_data_features.drop(columns='label').to_numpy(dtype=np.float64)
                    
                    y = can_data_features['label'].to_numpy()

                    clf = None
                    if classifier == 'log_regression':
                        clf = LogisticRegression(
                        penalty='l1', solver='saga', max_iter=1000, tol=1e-1, class_weight='balanced')
                    elif classifier == 'random_forest':
                        clf = XGBClassifier(objective='binary:hinge', n_estimators=150, scale_pos_weight=np.bincount(y)[0] / float(np.bincount(y)[1]), use_label_encoder=False, n_jobs=1)
                    else:
                        raise ValueError('Received unknown classifier string!')
                    
                    max_features = 16
                    best_score = 0
                    best_features = []
                    best_X = None
                    for n_features in range(10, max_features+1):
                        sfs = SequentialFeatureSelector(clf, k_features=n_features, scoring='balanced_accuracy', cv=LOGO, n_jobs=len(subject_ids))
                        X_new = sfs.fit_transform(StandardScaler().fit_transform(X, y), y, groups=groups)
                        score = sfs.k_score_
                        print('score with {} features: {}'.format(n_features, score))
                        if score > best_score:
                            best_score = score
                            best_features = can_data_features_bin.columns.to_numpy()[list(sfs.k_feature_idx_)]
                            best_X = X_new
                    print('best score (with {} features): {}'.format(len(best_features), best_score))
                    print(best_features)
                    # vt = VarianceThreshold(threshold=0.2)
                    # sfm = SelectFromModel(clf, threshold=-np.inf, max_features=50)

                    pipeline = make_pipeline(StandardScaler(), clf)

                    cv = cross_validate(estimator=pipeline, X=best_X, y=y, scoring=SCORING, return_estimator=True, verbose=0,
                            return_train_score=True, cv=LOGO, groups=groups, n_jobs=len(subject_ids))


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
                    
                    results = pd.DataFrame({k:v for k,v in cv.items() if k not in ['estimator']}).set_index(subject_ids)
                    mean = results.mean(axis=0).rename('mean')
                    std = results.std(axis=0).rename('stddev')
                    results = results.append(mean)
                    results = results.append(std)
                    results.to_csv(
                            'out/results/{}_pred_results_windowsize_{}_step_size_{}s{}_{}.csv'.format(
                                classifier, window_size, step, signal_string, scenario
                                ), index=True, header=True
                            )


def do_event_classification(classifier):
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

    can_data_events.loc[:, 'label'] = 0
    can_data_events.loc[(slice(None), 'above', slice(None), slice(None)), 'label'] = 1

    can_data_events.dropna(axis=1, inplace=True)

    # drop below BAC level for binary classification
    can_data_events.drop('below', level=1, inplace=True)


    for k, scenario in enumerate(SCENARIOS):
        print('scenario: {}'.format(scenario))

        can_data_events_bin = can_data_events.loc[:, :, scenario, :]

        groups = list(can_data_events_bin.index.get_level_values('subject_id'))
        subject_ids = np.unique(groups)
        
        X = can_data_events_bin.drop(columns='label').to_numpy(dtype=np.float64)
        
        y = can_data_events_bin['label'].to_numpy()

        clf = None
        if classifier == 'log_regression':
            clf = LogisticRegression(
            penalty='l1', solver='saga', max_iter=1000, tol=1e-1, class_weight='balanced')
        elif classifier == 'random_forest':
            clf = RandomForestClassifier(n_estimators=500, class_weight='balanced')
        else:
            raise ValueError('Received unknown classifier string!')
        
        #sfs = SequentialFeatureSelector(clf, n_features_to_select=5, scoring='roc_auc', n_jobs=len(subject_ids))
        #vt = VarianceThreshold(threshold=0.2)
        #sfm = SelectFromModel(clf, threshold=-np.inf, max_features=50)

        pipeline = make_pipeline(StandardScaler(), clf)

        cv = cross_validate(estimator=pipeline, X=X, y=y, scoring=SCORING, return_estimator=True, verbose=0,
                return_train_score=True, cv=LOGO, groups=groups, n_jobs=len(subject_ids))

        results = pd.DataFrame({k:v for k,v in cv.items() if k not in ['estimator']}).set_index(subject_ids)
        mean = results.mean(axis=0).rename('mean')
        std = results.std(axis=0).rename('stddev')
        results = results.append(mean)
        results = results.append(std)
        results.to_csv(
                'out/results/{}_pred_results_events_{}.csv'.format(
                    classifier, scenario), index=True, header=True
                )