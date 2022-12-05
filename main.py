from helpers import ConfigLoader
from train import TrainingPipeline
from generate_plots import PlotCreator

if __name__ == "__main__":
    config_dict = ConfigLoader.setup_config("config.yaml")
    pipeline = TrainingPipeline(config_dict)
    pipeline.initialize()
    pipeline.run_trials()
    plot_creator = PlotCreator(config_dict, pipeline.results_dict)
    plot_creator.generate_plots()
    plot_creator.create_confusion_matrix()

