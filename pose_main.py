from helpers import ConfigLoader
from train import TrainingPipeline

import numpy as np
import matplotlib.pyplot as plt

def save_results(config_dict, pipeline, results_dict):
    model_name = config_dict['network_name']
    root = "results/"

    train_loss = np.array(results_dict["train_loss"])
    np.save(f"{root}{model_name}_train_loss.npy", train_loss)
    
    val_loss = np.array(results_dict["val_loss"])
    np.save(f"{root}{model_name}_val_loss.npy", val_loss)

    train_acc = np.array(results_dict["train_acc"])
    np.save(f"{root}{model_name}_train_acc.npy", train_acc)
    
    val_acc = np.array(results_dict["val_acc"])
    np.save(f"{root}{model_name}_val_acc.npy", val_acc)

    epoch_num = np.array(results_dict["epoch_num"])
    np.save(f"{root}{model_name}_epoch_num.npy", epoch_num)

    

def create_plots(config_dict):
    model_name = config_dict['network_name']
    root = "results/"
    epochs = np.load(f"{root}{model_name}_epoch_num.npy")
    train_loss = np.load(f"{root}{model_name}_train_loss.npy")
    val_loss = np.load(f"{root}{model_name}_val_loss.npy")
    train_acc = np.load(f"{root}{model_name}_train_acc.npy")
    val_acc = np.load(f"{root}{model_name}_val_acc.npy")
    
    # plot accuracy
    plt.plot(epochs, train_acc, label="train accuracy")
    plt.plot(epochs, val_acc, label="val accuracy")

    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy over epochs')
    plt.legend()
    plt.savefig(f"{root}/accuracy_{model_name}.png")
    plt.clf()
    # plt.close()

    # plot loss 
    plt.plot(epochs, train_loss, label="train loss")
    plt.plot(epochs, val_loss, label="val loss")
    plt.legend()
    plt.savefig(f"{root}/loss_{model_name}.png")
    

if __name__ == "__main__":
    config_dict = ConfigLoader.setup_config("config.yaml")
    create_plots(config_dict)
    pipeline = TrainingPipeline(config_dict)
    pipeline.initialize()
    pipeline.run_trials()
    results_dict = pipeline.results[0]
    save_results(config_dict, pipeline, results_dict)
