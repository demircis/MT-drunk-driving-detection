import pandas as pd
from classifier import Classifier

SCORING = ['balanced_accuracy', 'roc_auc', 'f1_micro', 'average_precision', 'recall', 'precision']

SIGNAL_COMBOS = [['driver_behavior', 'vehicle_behavior'], ['driver_behavior', 'vehicle_behavior', 'navi'],
                ['driver_behavior', 'vehicle_behavior', 'radar'], ['driver_behavior', 'vehicle_behavior', 'navi', 'radar']]

EVENTS = ['brake', 'brake_to_gas', 'gas', 'gas_to_brake', 'overtaking', 'road_sign', 'turning']

SELECTED_SIGNALS = {
    'driver_behavior': [
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
        'SteerSpeed_jerk'
    ],
    'vehicle_behavior': [
        'acc',
        'acc_jerk',
        'velocity',
        'latvel_acc',
        'latvel_jerk',
        'YawRate_acc',
        'YawRate_jerk',
        'YawRate'
    ],
    'navi': [
        'speed_limit_exceeded',
        'SpeedDif'
    ],
    'radar': [
        'Dhw',
        'is_crossing_lane_left',
        'is_crossing_lane_right',
        'lane_crossing',
        'lane_distance_left_edge',
        'lane_distance_right_edge',
        'Ttc',
        'TtcOpp'
    ]
}

STATS = ['mean', 'std', 'min', 'max', 'q5', 'q95', 'iqrange', 'iqrange_5_95', 'skewness', 'kurtosis', 'peaks', 'rms']

SUM_COLUMNS = ['lane_crossing', 'lane_crossing_left', 'lane_crossing_right', 'is_crossing_lane', 'is_crossing_lane_left', 'is_crossing_lane_right', 'speed_limit_exceeded']

SCENARIOS = ['highway', 'rural', 'town']

def do_sliding_window_classification(window_sizes, classifier_type, classifier_mode, per_scenario):
    for window_size in window_sizes:
        for combo in SIGNAL_COMBOS:
            signal_string = ''
            can_data_features = []
            for signal in combo:
                signal_string += '_' + signal
                can_data_features.append(
                    pd.read_parquet('out/can_data_features_{}_windowsize_{}s.parquet'.format(signal, window_size), columns=select_columns(signal))
                )
            can_data_features = pd.concat(can_data_features, axis=1)
            
            if classifier_type == 'log_regression':
                can_data_features.dropna(axis=1, inplace=True)

            clf = Classifier(classifier_type, classifier_mode, max_features=50)
            if per_scenario:
                for scenario in SCENARIOS:
                    print('signals: {}, window size: {}s, scenario: {}'.format(
                        signal_string, window_size, scenario
                        ))
                    
                    can_data_features_scenario = can_data_features.loc[:, :, scenario, :]

                    results, selected_features = clf.do_classification(can_data_features_scenario, scenario=scenario)
                    selected_features.to_csv(
                            'out/results/{}_{}_selected_features_windowsize_{}{}_{}.csv'.format(
                                classifier_type, classifier_mode, window_size, signal_string, scenario
                                ), index=True, header=['selected_features']
                            )

                    results.to_csv(
                            'out/results/{}_{}_pred_results_windowsize_{}{}_{}.csv'.format(
                                classifier_type, classifier_mode, window_size, signal_string, scenario
                                ), index=True, header=True
                            )
            else:
                print('signals: {}, window size: {}s'.format(signal_string, window_size))
                    
                results, selected_features = clf.do_classification(can_data_features)
                selected_features.to_csv(
                        'out/results/{}_{}_selected_features_windowsize_{}{}.csv'.format(
                            classifier_type, classifier_mode, window_size, signal_string
                            ), index=True, header=['selected_features']
                        )

                results.to_csv(
                        'out/results/{}_{}_pred_results_windowsize_{}{}.csv'.format(
                            classifier_type, classifier_mode, window_size, signal_string
                            ), index=True, header=True
                        )


def do_combined_events_classification(classifier_type, classifier_mode, per_scenario):
    can_data_events = []
    for event in EVENTS:
        can_data_events.append(
            pd.read_parquet('out/can_data_{}_events_features.parquet'.format(event), columns=['duration'] + select_columns())
        )
    can_data_events = pd.concat(can_data_events, axis=0)

    if classifier_type == 'log_regression':
        can_data_events.dropna(axis=1, inplace=True)

    clf = Classifier(classifier_type, classifier_mode, max_features=20)
    if per_scenario:
        for scenario in SCENARIOS:
            print('scenario: {}'.format(scenario))

            can_data_events_scenario = can_data_events.loc[:, :, scenario, :]
            
            results, selected_features = clf.do_classification(can_data_events_scenario, scenario)
            selected_features.to_csv(
                    'out/results/{}_{}_selected_features_combined_events_{}.csv'.format(
                        classifier_type, classifier_mode, scenario
                        ), index=True, header=['selected_features']
                    )

            results.to_csv(
                    'out/results/{}_{}_pred_results_combined_events_{}.csv'.format(
                        classifier_type, classifier_mode, scenario), index=True, header=True
                    )
    else:
        results, selected_features = clf.do_classification(can_data_events)
        selected_features.to_csv(
                'out/results/{}_{}_selected_features_combined_events.csv'.format(
                    classifier_type, classifier_mode
                    ), index=True, header=['selected_features']
                )

        results.to_csv(
                'out/results/{}_{}_pred_results_combined_events.csv'.format(
                    classifier_type, classifier_mode), index=True, header=True
                )


def do_per_event_classification(classifier_type, classifier_mode, per_scenario):
    for event in EVENTS:
        can_data_events = pd.read_parquet('out/can_data_{}_events_features.parquet'.format(event), columns=['duration'] + select_columns())

        if classifier_type == 'log_regression':
            can_data_events.dropna(axis=1, inplace=True)

        clf = Classifier(classifier_type, classifier_mode, max_features=20)

        if per_scenario:
            for scenario in SCENARIOS:
                print('event type: {}, scenario: {}'.format(event, scenario))

                if event == 'turning' and scenario == 'highway':
                    continue

                can_data_events_scenario = can_data_events.loc[:, :, scenario, :]

                results, selected_features = clf.do_classification(can_data_events_scenario, scenario)
                selected_features.to_csv(
                        'out/results/{}_{}_selected_features_{}_{}.csv'.format(
                            classifier_type, classifier_mode, event, scenario
                            ), index=True, header=['selected_features']
                        )

                results.to_csv(
                        'out/results/{}_{}_pred_results_{}_{}.csv'.format(
                            classifier_type, classifier_mode, event, scenario), index=True, header=True
                        )
        else:
            print('event type: {}'.format(event))

            results, selected_features = clf.do_classification(can_data_events)
            selected_features.to_csv(
                    'out/results/{}_{}_selected_features_{}.csv'.format(
                        classifier_type, classifier_mode, event
                        ), index=True, header=['selected_features']
                    )

            results.to_csv(
                    'out/results/{}_{}_pred_results_{}.csv'.format(
                        classifier_type, classifier_mode, event), index=True, header=True
                    )


def do_events_sliding_window_classification(window_sizes, classifier_type, classifier_mode, per_scenario):
    for window_size in window_sizes:
        can_data_events_per_window = []
        for event in EVENTS:
            cols = [event + '_event_' + col + '-' + stat for stat in ['mean', 'std', 'q5', 'q95', 'skewness', 'kurtosis'] for col in ['duration'] + select_columns()]
            can_data_events_per_window.append(pd.read_parquet(
                'out/can_data_{}_events_per_window_windowsize_{}s.parquet'.format(event, window_size), columns=[event + '_event_ratio', event + '_event_count'] + cols
                ))
        can_data_events_per_window = pd.concat(can_data_events_per_window, axis=1)

        for event in EVENTS:
            cols = [event + '_event_' + col + '-' + stat for stat in ['mean', 'std', 'q5', 'q95', 'skewness', 'kurtosis'] for col in ['duration'] + select_columns()]
            can_data_events_per_window.loc[can_data_events_per_window[cols].isna().all(axis=1), cols] = 0

        if classifier_type == 'log_regression':
                can_data_events_per_window.dropna(axis=1, inplace=True)

        clf = Classifier(classifier_type, classifier_mode, max_features=20)
        
        if per_scenario:
            for scenario in SCENARIOS:
                print('window size: {}s, scenario: {}'.format(window_size, scenario))
                
                can_data_events_per_window_scenario = can_data_events_per_window.loc[:, :, scenario, :]

                results, selected_features = clf.do_classification(can_data_events_per_window_scenario, scenario)

                selected_features.to_csv(
                        'out/results/{}_{}_selected_features_events_per_window_windowsize_{}_{}.csv'.format(
                            classifier_type, classifier_mode, window_size, scenario
                            ), index=True, header=['selected_features']
                        )

                results.to_csv(
                        'out/results/{}_{}_pred_results_events_per_window_windowsize_{}_{}.csv'.format(
                            classifier_type, classifier_mode, window_size, scenario
                            ), index=True, header=True
                        )
        else:
            print('window size: {}s'.format(window_size))

            results, selected_features = clf.do_classification(can_data_events_per_window)
            selected_features.to_csv(
                    'out/results/{}_{}_selected_features_events_per_window_windowsize_{}.csv'.format(
                        classifier_type, classifier_mode, window_size
                        ), index=True, header=['selected_features']
                    )

            results.to_csv(
                    'out/results/{}_{}_pred_results_events_per_window_windowsize_{}.csv'.format(
                        classifier_type, classifier_mode, window_size
                        ), index=True, header=True
                    )


def do_combined_classification(window_sizes, classifier_type, classifier_mode, per_scenario):
    for window_size in window_sizes:
        for combo in SIGNAL_COMBOS:
            signal_string = ''
            can_data_features = []
            for signal in combo:
                signal_string += '_' + signal
                can_data_features.append(
                    pd.read_parquet('out/can_data_features_{}_windowsize_{}s.parquet'.format(signal, window_size), columns=['duration'] + select_columns(signal))
                )
            can_data_features = pd.concat(can_data_features, axis=1)

            can_data_features = can_data_features.loc[:,~can_data_features.columns.duplicated()]
            
            can_data_events = []
            for event in EVENTS:
                can_data_events.append(
                    pd.read_parquet('out/can_data_{}_events_features.parquet'.format(event), columns=['duration'] + select_columns())
                    )
            can_data_events = pd.concat(can_data_events, axis=0)

            can_data_combined = pd.concat((can_data_features, can_data_events), axis=0)

            if classifier_type == 'log_regression':
                can_data_combined.dropna(axis=1, inplace=True)

            clf = Classifier(classifier_type, classifier_mode, max_features=50)
            if per_scenario:
                for scenario in SCENARIOS:
                    print('signals: {}, window size: {}s, scenario: {}'.format(
                        signal_string, window_size, scenario
                        ))
                    
                    can_data_combined_scenario = can_data_combined.loc[:, :, scenario, :]

                    results, selected_features = clf.do_classification(can_data_combined_scenario, scenario)
                    selected_features.to_csv(
                            'out/results/{}_{}_selected_features_combined_windowsize_{}{}_{}.csv'.format(
                                classifier_type, classifier_mode, window_size, signal_string, scenario
                                ), index=True, header=['selected_features']
                            )

                    results.to_csv(
                            'out/results/{}_{}_pred_results_combined_windowsize_{}{}_{}.csv'.format(
                                classifier_type, classifier_mode, window_size, signal_string, scenario
                                ), index=True, header=True
                            )
            else:
                print('signals: {}, window size: {}s'.format(signal_string, window_size))
                
                results, selected_features = clf.do_classification(can_data_combined_scenario, scenario)
                selected_features.to_csv(
                        'out/results/{}_{}_selected_features_combined_windowsize_{}{}.csv'.format(
                            classifier_type, classifier_mode, window_size, signal_string
                            ), index=True, header=['selected_features']
                        )

                results.to_csv(
                        'out/results/{}_{}_pred_results_combined_windowsize_{}{}.csv'.format(
                            classifier_type, classifier_mode, window_size, signal_string
                            ), index=True, header=True
                        )


def do_signal_combo_classification(classifier_type, classifier_mode):
    window_size = 60
    for combo in SIGNAL_COMBOS:
        signal_string = ''
        can_data_features = []
        for signal in combo:
            signal_string += '_' + signal
            can_data_features.append(
                pd.read_parquet('out/can_data_features_{}_windowsize_{}s.parquet'.format(signal, window_size), columns=select_columns(signal))
                )
        can_data_features = pd.concat(can_data_features, axis=1)

        print('signals: {}'.format(signal_string))

        if classifier_type == 'log_regression':
            can_data_features.dropna(axis=1, inplace=True)

        clf = Classifier(classifier_type, classifier_mode, max_features=None)

        results, _ = clf.do_classification(can_data_features)

        results.to_csv(
                'out/results/{}_{}_pred_results_windowsize_{}{}.csv'.format(
                    classifier_type, classifier_mode, window_size, signal_string
                    ), index=True, header=True
                )


def do_window_size_classification(window_sizes, classifier_type, classifier_mode):
    for window_size in window_sizes:
        print('window size: {}s'.format(window_size))
        signal_string = ''
        can_data_features = []
        for signal in ['driver_behavior', 'vehicle_behavior', 'navi', 'radar']:
            signal_string += '_' + signal
            can_data_features.append(
                pd.read_parquet('out/can_data_features_{}_windowsize_{}s.parquet'.format(signal, window_size), columns=select_columns(signal))
            )
        can_data_features = pd.concat(can_data_features, axis=1)
        
        if classifier_type == 'log_regression':
            can_data_features.dropna(axis=1, inplace=True)

        clf = Classifier(classifier_type, classifier_mode, max_features=None)
                
        results, _ = clf.do_classification(can_data_features)

        results.to_csv(
                'out/results/{}_{}_pred_results_windowsize_{}.csv'.format(
                    classifier_type, classifier_mode, window_size
                    ), index=True, header=True
                )


def do_overlap_percentage_classification(overlap_percentages, classifier_type, classifier_mode):
    window_size = 60
    for overlap_percentage in overlap_percentages:
        signal_string = ''
        can_data_features = []
        for signal in ['driver_behavior', 'vehicle_behavior', 'navi', 'radar']:
            signal_string += '_' + signal
            can_data_features.append(
                pd.read_parquet('out/can_data_features_{}_windowsize_{}s.parquet'.format(signal, window_size), columns=select_columns(signal))
                )
        can_data_features = pd.concat(can_data_features, axis=1)
        
        if classifier_type == 'log_regression':
            can_data_features.dropna(axis=1, inplace=True)

        step = 1
        if overlap_percentage is not None:
            step = window_size - int(overlap_percentage * window_size)
        can_data_features_step = can_data_features[(can_data_features.groupby(['subject_id', 'subject_state', 'subject_scenario']).cumcount() % step) == 0]

        clf = Classifier(classifier_type, classifier_mode, max_features=None)
        print('window size: {}s'.format(window_size))
                
        results, _ = clf.do_classification(can_data_features_step)

        results.to_csv(
                'out/results/{}_{}_pred_results_step_size_{}_windowsize_{}.csv'.format(
                    classifier_type, classifier_mode, step, window_size
                    ), index=True, header=True
                )


def select_columns(signals=None):
    all_signals = [signal for signal_type in list(SELECTED_SIGNALS.values()) for signal in signal_type]
    stat_columns_list = [
        [sig + '_' + s for s in (['mean', 'std', 'sum'] if sig in SUM_COLUMNS else STATS)]
        for sig in (SELECTED_SIGNALS[signals] if signals is not None else all_signals)]
    stat_columns = []
    for item in stat_columns_list:
        stat_columns += item
    return stat_columns
