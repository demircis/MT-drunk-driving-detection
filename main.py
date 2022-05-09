from yaml import load, Loader
from classification import do_combined_classification, do_combined_events_classification, do_events_sliding_window_classification, do_overlap_percentage_classification, do_signal_combo_classification, do_sliding_window_classification, do_per_event_classification, do_window_size_classification
from generate_can_features import calc_can_data_event_features, calc_can_data_features, calc_event_features_in_window
from parse_scenario_information import parse_scenario_information
from preprocess_can_data import do_preprocessing

from bunch import Bunch

if __name__ == '__main__':
    stream = open("config.yaml", 'r')
    config = Bunch(load(stream, Loader=Loader))

    if config.parse_scenario_info:
        parse_scenario_information()

    if config.preprocess:
        do_preprocessing(config.full_study, config.data_freq)

    if config.generate_features:
        calc_can_data_features(config.window_sizes)
        calc_can_data_event_features()
        calc_event_features_in_window(config.window_sizes)

    if config.do_classification:
        if config.dataset == 'sliding_window':
            do_sliding_window_classification(config.window_sizes, config.classifier_type, config.clf_mode, config.per_scenario)
        elif config.dataset == 'events':
            do_combined_events_classification(config.classifier_type, config.clf_mode, config.per_scenario)
            do_per_event_classification(config.classifier_type, config.clf_mode, config.per_scenario)
        elif config.dataset == 'events_sliding_window':
            do_events_sliding_window_classification(config.window_sizes, config.classifier_type, config.clf_mode, config.per_scenario)
        elif config.dataset == 'combined':
            do_combined_classification(config.window_sizes, config.classifier_type, config.clf_mode, config.per_scenario)
        elif config.dataset == 'window_sizes':
            do_window_size_classification(config.window_sizes, config.classifier_type, config.clf_mode)
        elif config.dataset == 'signal_combos':
            do_signal_combo_classification(config.classifier_type, config.clf_mode)
        elif config.dataset == 'overlap_percentages':
            do_overlap_percentage_classification(config.overlap_percentages, config.classifier_type, config.clf_mode)
