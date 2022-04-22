from yaml import load, Loader
from classification import do_combined_classification, do_sliding_window_classification, do_event_classification
from generate_can_features import calc_can_data_features, calc_event_features_in_window, filter_can_data_event_columns
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
        filter_can_data_event_columns()
        calc_event_features_in_window(config.window_sizes)

    if config.do_classification:
        if config.dataset == 'sliding_window':
            do_sliding_window_classification(config.window_sizes, config.overlap_percentages, config.classifier, config.clf_mode)
        elif config.dataset == 'events':
            do_event_classification(config.classifier, config.clf_mode)
        elif config.dataset == 'combined':
            do_combined_classification(config.classifier, config.window_sizes, config.clf_mode)
