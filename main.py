from helpers import ConfigLoader
from train import TrainingPipeline

if __name__ == "__main__":
    config_dict = ConfigLoader.setup_config("config.yaml")
    pipeline = TrainingPipeline(config_dict)
    pipeline.initialize()
    pipeline.run_trials()