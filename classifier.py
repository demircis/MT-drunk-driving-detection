import pandas as pd
import numpy as np
from sklearnex import patch_sklearn
patch_sklearn()
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import cross_validate
from sklearn.model_selection import LeaveOneGroupOut

class Classifier:
    RANDOM_STATE = 42

    SCORING = ('balanced_accuracy', 'roc_auc', 'f1_micro', 'average_precision', 'recall', 'precision')

    SIGNAL_COMBOS = (('driver_behavior', 'vehicle_behavior'), ('driver_behavior', 'vehicle_behavior', 'navi'),
                    ('driver_behavior', 'vehicle_behavior', 'radar'), ('driver_behavior', 'vehicle_behavior', 'navi', 'radar'))

    EVENTS = ('brake', 'brake_to_gas', 'gas', 'gas_to_brake', 'overtaking', 'road_sign', 'turning')

    SELECTED_SIGNALS = (
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
    )

    STATS = ('mean', 'std', 'min', 'max', 'q5', 'q95', 'iqrange', 'iqrange_5_95', 'skewness', 'kurtosis', 'peaks', 'rms')

    SUM_COLUMNS = ('lane_crossing', 'lane_crossing_left', 'lane_crossing_right', 'is_crossing_lane', 'is_crossing_lane_left', 'is_crossing_lane_right', 'speed_limit_exceeded')

    SCENARIOS = ('highway', 'rural', 'town')

    LOGO = LeaveOneGroupOut()

    def __init__(self, classifier_type, classifier_mode, max_features):
        self.classifier_type = classifier_type
        self.classifier_mode = classifier_mode
        self.max_features = max_features
        self.estimator = None

        if self.classifier_mode == 'binary':
            if self.classifier_type == 'log_regression':
                self.estimator = LogisticRegression(
                penalty='l1', solver='saga', max_iter=1000, tol=1e-2, random_state=self.RANDOM_STATE)
            elif self.classifier_type == 'random_forest':
                self.estimator = LGBMClassifier(objective='binary', n_estimators=100, n_jobs=1, random_state=self.RANDOM_STATE)
            else:
                raise ValueError('Received unknown classifier string!')
        elif self.classifier_mode == 'multiclass':
            if self.classifier_type == 'log_regression':
                self.estimator = LogisticRegression(
                penalty='l1', solver='saga', max_iter=1000, tol=1e-2, random_state=self.RANDOM_STATE)
            elif self.classifier_type == 'random_forest':
                self.estimator = LGBMClassifier(objective='multiclass', n_estimators=100, n_jobs=1, random_state=self.RANDOM_STATE)
            else:
                raise ValueError('Received unknown classifier string!')
    

    def do_classification(self, data, scenario=None):                
        X, y, weights, groups = self.prepare_dataset(data, scenario)
        
        subject_ids = np.unique(groups)

        best_X = None
        selected_features = None
        if self.max_features is not None:
            best_X, selected_features = self.sequential_feature_selection(X, y, groups, weights, len(subject_ids)-1)
        else:
            best_X = X

        cv = cross_validate(estimator=self.estimator, X=best_X, y=y, scoring=self.SCORING, return_estimator=True, verbose=0,
                return_train_score=True, cv=self.LOGO, groups=groups, n_jobs=len(subject_ids)-1, fit_params={'sample_weight': weights})
        
        results = self.collect_results(cv, subject_ids)
        return results, selected_features
    

    def prepare_dataset(self, data, scenario=None):
        input_data = data.copy()
        input_data.loc[:, 'label'] = 0
        if self.classifier_mode == 'binary':
            input_data.loc[(slice(None), 'above', slice(None), slice(None)), 'label'] = 1
            # drop below BAC level for binary classification
            input_data.drop('below', level=1, inplace=True)
        elif self.classifier_mode == 'multiclass':
            input_data.loc[(slice(None), 'below', slice(None), slice(None)), 'label'] = 1
            input_data.loc[(slice(None), 'above', slice(None), slice(None)), 'label'] = 2
        else:
            raise ValueError('Received unknown classifier mode string!')
        
        if scenario == 'highway':
            input_data.drop(columns=list(input_data.filter(like = 'TtcOpp')), inplace=True, errors='ignore')
            input_data.drop(columns=list(input_data.filter(like = 'brake')), inplace=True, errors='ignore')
        X = input_data.drop(columns='label')
        
        y = input_data['label']
        class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y.to_numpy())
        weights = np.zeros(y.shape)
        for i, weight in enumerate(class_weights):
            weights[y == i] = weight

        groups = list(input_data.index.get_level_values('subject_id'))

        X = pd.DataFrame(StandardScaler().fit_transform(X.to_numpy(), y.to_numpy()), columns=list(X.columns))

        return X, y, weights, groups


    def sequential_feature_selection(self, X, y, groups=None, weights=None, n_jobs=1):
        sfs = SequentialFeatureSelector(self.estimator, k_features=(1, self.max_features), scoring='balanced_accuracy', cv=self.LOGO, n_jobs=n_jobs, verbose=2)
        best_X = sfs.fit_transform(X, y, groups=groups, sample_weight=weights)
        print('\nbest score (with {} features): {}'.format(len(list(sfs.k_feature_names_)), sfs.k_score_))
        selected_features = pd.Series(list(sfs.k_feature_names_))
        print(selected_features.to_numpy())
        return best_X, selected_features


    def collect_results(self, cv_object, subject_ids):
        results = pd.DataFrame({k:v for k,v in cv_object.items() if k not in ['estimator']}).set_index(subject_ids)
        mean = results.mean(axis=0)
        std = results.std(axis=0)
        results = pd.concat((results, mean.to_frame('mean').T))
        results = pd.concat((results, std.to_frame('stddev').T))
        return results

