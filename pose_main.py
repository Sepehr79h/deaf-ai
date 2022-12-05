from helpers import ConfigLoader
from train import TrainingPipeline
from generate_plots import PlotCreator

if __name__ == "__main__":
    config_dict = ConfigLoader.setup_config("config.yaml")
    pipeline = TrainingPipeline(config_dict)
    pipeline.initialize()
    pipeline.run_trials()
    print("~~~ Generating Plots ~~~")
    plot_creator = PlotCreator(config_dict, pipeline.results)
    plot_creator.generate_plots()


