from yaml import load, Loader
from generate_can_features import store_can_data_features
from parse_scenario_information import parse_scenario_information
from preprocess_can_data import do_preprocessing

from bunch import Bunch

if __name__ == '__main__':
    stream = open("config.yaml", 'r')
    config = Bunch(load(stream, Loader=Loader))

    if config.parse_scenario_info:
        parse_scenario_information()

    if config.preprocess and config.overwrite:
        do_preprocessing(config.full_study, config.overwrite, config.data_freq)

    if config.generate_features:
        store_can_data_features(config.window_sizes)
