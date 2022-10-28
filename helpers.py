import sys
import yaml


class ConfigLoader:
    @staticmethod
    def setup_config(path_to_config):
        """
        Reads config.yaml to set hyperparameters
        """

        with open(path_to_config, 'r') as stream:
            try:
                config = yaml.safe_load(stream)
                return config
            except yaml.YAMLError as exc:
                print(exc)
                print("Could not read from config.yaml, exiting...")
                sys.exit(1)
