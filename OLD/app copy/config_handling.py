import configparser
import os


def read_config():
    # Get the path to the configuration file within the module
    module_dir = os.path.dirname(__file__)
    config_file_path = os.path.join(module_dir, "config.cfg")
    # Initialize configparser
    config = configparser.ConfigParser()
    config.read(config_file_path)
    return config
