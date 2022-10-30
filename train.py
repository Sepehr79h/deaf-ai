import time
import torch.nn

from dataset import *
from models import *
import matplotlib.pyplot as plt
import numpy as np


class TrainingPipeline:
    def __init__(self, config_dict):
        self.config_dict = config_dict
        self.epochs = int(self.config_dict["num_epochs"])
        self.total_trials = int(self.config_dict["total_trials"])
        self.data_creator = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.device = None
        self.optimizer = None
        self.loss = None
        self.model = None
        self.results = None

    def initialize(self):
        optimizer_dict = {"Adam": torch.optim.Adam, "SGD": torch.optim.SGD}
        self.optimizer = optimizer_dict.get(self.config_dict["optimizer"])
        if self.optimizer is None:
            raise AssertionError(f'Optimizer must be one of the following: {list(optimizer_dict.keys())}')
        self.data_creator = DataCreator(self.config_dict)
        self.train_loader, self.val_loader, self.test_loader = self.data_creator.create_loaders()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.loss = torch.nn.BCELoss()
        self.model = NetworkRetriever.get_network(self.config_dict["network_name"])

    def run_trials(self):
        self.results = {}
        for trial in range(self.total_trials):
            self.results[trial] = self.train(trial)
            print("-------------------")
            print(f"Best train accuracy for Trial {trial} is {max(self.results[trial]['train_acc'])}")
            print(f"Best val accuracy for Trial {trial} is {max(self.results[trial]['val_acc'])}")
            print(f"Final test accuracy for Trial {trial} is {self.results[trial]['test_acc']}")
            print("-------------------")

    def train(self, trial):
        true_start = time.time()
        results_dict = dict(epoch_num=[], epoch_times=[], train_loss=[], train_acc=[], val_loss=[], val_acc=[])
        for epoch in range(self.epochs):
            start_time = time.time()
            train_loss, train_accuracy, val_loss, val_accuracy = \
                self.epoch_iteration()

            end_time = time.time()
            print(
                f"Trial {trial}/{self.total_trials - 1} | " +
                f"Epoch {epoch}/{self.epochs} Ended | " +
                "Total Time: {:.3f}s | ".format(end_time - true_start) +
                "Epoch Time: {:.3f}s | ".format(end_time - start_time) +
                "~Time Left: {:.3f}s | ".format(
                    (end_time - start_time) * (self.epochs - epoch + 1)),
                "Train Loss: {:.4f}% | Train Acc. {:.4f}% | ".format(
                    train_loss,
                    train_accuracy) +
                "Val Loss: {:.4f}% | Val Acc. {:.4f}%".format(val_loss,
                                                              val_accuracy))
            results_dict["epoch_num"].append(epoch)
            results_dict["epoch_times"].append(round(end_time - start_time, 3))
            results_dict["train_loss"].append(round(train_loss, 4))
            results_dict["val_loss"].append(round(val_loss, 4))
            results_dict["train_acc"].append(round(train_accuracy, 4))
            results_dict["val_acc"].append(round(val_accuracy, 4))
        test_loss, test_accuracy = self.evaluate(self.val_loader, self.device)
        results_dict["test_loss"] = test_loss
        results_dict["test_acc"] = test_accuracy
        return results_dict

    def evaluate(self, eval_loader, device):
        self.model.eval()
        eval_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(eval_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = self.model(inputs)
                loss = self.loss(outputs, targets)

                eval_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        acc = 100. * correct / total
        return eval_loss / (batch_idx + 1), acc

    def epoch_iteration(self):
        self.model.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss(outputs, targets)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        val_loss, val_accuracy = self.evaluate(self.val_loader, self.device)

        return (train_loss / (len(self.train_loader) + 1), 100. * correct / total,
                val_loss, val_accuracy)
    
    def plot_accuracy(self, epochs, train_accuracy, validation_accuracy=None, test_accuracy=None, test_epochs=None):
        if type(epochs == int):
            epochs = np.arange(0, epochs, 1)
        plt.plot(epochs, train_accuracy, label="train accuracy")
        if type(validation_accuracy) != None:
            plt.plot(epochs, validation_accuracy, label="validation accuracy")

        plt.ylabel('Accuracy (%)')
        plt.title('Accuracy over epochs')
        plt.legend()
        plt.show()

        plt.close()

        if type(test_accuracy) == None or type(test_epochs) == None:
            print("skipping test plotting because either test accuarcy or test epochs is None")
            return

        if type(test_epochs == int):
            test_epochs = np.arange(0, test_epochs, 1)
        plt.plot(test_epochs, test_accuracy, label="test accuracy")

        plt.ylabel('Accuracy (%)')
        plt.title('Accuracy over epochs (Test)')
        plt.legend()
        plt.show()
