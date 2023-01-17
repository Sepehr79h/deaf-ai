# import time
# import torch.nn
#
# from dataset import *
# from models import *
# from torchsummary import summary
#
#
# class TrainingPipeline:
#     def __init__(self, config_dict):
#         self.config_dict = config_dict
#         self.epochs = int(self.config_dict["num_epochs"])
#         self.total_trials = int(self.config_dict["total_trials"])
#         self.data_creator = None
#         self.train_loader = None
#         self.val_loader = None
#         self.test_loader = None
#         self.device = None
#         self.optimizer = None
#         self.loss = None
#         self.model = None
#         self.results = None
#
#     def initialize(self):
#         optimizer_dict = {"Adam": torch.optim.Adam, "SGD": torch.optim.SGD}
#         self.optimizer = optimizer_dict.get(self.config_dict["optimizer"])
#         if self.optimizer is None:
#             raise AssertionError(f'Optimizer must be one of the following: {list(optimizer_dict.keys())}')
#         self.data_creator = PoseDatasetCreator(self.config_dict)
#         self.train_loader, self.val_loader, self.test_loader = self.data_creator.create_loaders()
#         self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
#         self.loss = torch.nn.BCELoss()
#         self.model = NetworkRetriever.get_network(self.config_dict["network_name"])
#         self.optimizer = self.optimizer(self.model.parameters(), lr=float(self.config_dict["learning_rate"]))
#         #print(summary(self.model, input_size=(100, 38)))
#
#     def run_trials(self):
#         self.results = {}
#         for trial in range(self.total_trials):
#             self.results[trial] = self.train(trial)
#             print("-------------------")
#             print(f"Best train accuracy for Trial {trial} is {max(self.results[trial]['train_acc'])}")
#             print(f"Best val accuracy for Trial {trial} is {max(self.results[trial]['val_acc'])}")
#             print(f"Final test accuracy for Trial {trial} is {self.results[trial]['test_acc']}")
#             print("-------------------")
#
#     def train(self, trial):
#         true_start = time.time()
#         results_dict = dict(epoch_num=[], epoch_times=[], train_loss=[], train_acc=[], val_loss=[], val_acc=[])
#         for epoch in range(self.epochs):
#             start_time = time.time()
#             train_loss, train_accuracy, val_loss, val_accuracy = \
#                 self.epoch_iteration()
#
#             end_time = time.time()
#             print(
#                 f"Trial {trial}/{self.total_trials - 1} | " +
#                 f"Epoch {epoch}/{self.epochs} Ended | " +
#                 "Total Time: {:.3f}s | ".format(end_time - true_start) +
#                 "Epoch Time: {:.3f}s | ".format(end_time - start_time) +
#                 "~Time Left: {:.3f}s | ".format(
#                     (end_time - start_time) * (self.epochs - epoch + 1)),
#                 "Train Loss: {:.4f}% | Train Acc. {:.4f}% | ".format(
#                     train_loss,
#                     train_accuracy) +
#                 "Val Loss: {:.4f}% | Val Acc. {:.4f}%".format(val_loss,
#                                                               val_accuracy))
#             results_dict["epoch_num"].append(epoch)
#             results_dict["epoch_times"].append(round(end_time - start_time, 3))
#             results_dict["train_loss"].append(round(train_loss, 4))
#             results_dict["val_loss"].append(round(val_loss, 4))
#             results_dict["train_acc"].append(round(train_accuracy, 4))
#             results_dict["val_acc"].append(round(val_accuracy, 4))
#         test_loss, test_accuracy = self.evaluate(self.val_loader, self.device)
#         results_dict["test_loss"] = test_loss
#         results_dict["test_acc"] = test_accuracy
#         return results_dict
#
#     def evaluate(self, eval_loader, device):
#         self.model.eval()
#         eval_loss = 0
#         correct = 0
#         total = 0
#         with torch.no_grad():
#             for batch_idx, (inputs, targets) in enumerate(eval_loader):
#                 inputs = inputs.to(self.device).float()
#                 targets = targets.to(self.device).float().unsqueeze(1)
#                 outputs = self.model(inputs)
#                 loss = self.loss(outputs, targets)
#
#                 eval_loss += loss.item()
#                 _, predicted = outputs.max(1)
#                 total += targets.size(0)
#                 correct += predicted.eq(targets.squeeze(1)).sum().item()
#         acc = 100. * correct / total
#         return eval_loss / (batch_idx + 1), acc
#
#     def epoch_iteration(self):
#         self.model.train()
#         train_loss = 0
#         correct = 0
#         total = 0
#         for batch_idx, (inputs, targets) in enumerate(self.train_loader):
#             inputs = inputs.to(self.device).float()
#             targets = targets.to(self.device).float().unsqueeze(1)
#             self.optimizer.zero_grad()
#             outputs = self.model(inputs)
#             loss = self.loss(outputs, targets)
#             loss.backward()
#             self.optimizer.step()
#             #self.optimizer.zero_grad()
#
#             train_loss += loss.item()
#             _, predicted = outputs.max(1)
#             total += targets.size(0)
#             correct += predicted.eq(targets.squeeze(1)).sum().item()
#
#         val_loss, val_accuracy = self.evaluate(self.val_loader, self.device)
#
#         return (train_loss / (len(self.train_loader) + 1), 100. * correct / total,
#                 val_loss, val_accuracy)

import time
import torch.nn
from tqdm import tqdm

from dataset import *
from models import *


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
        self.data_creator = PoseDatasetCreator(self.config_dict)
        self.train_loader, self.val_loader, self.test_loader = self.data_creator.create_loaders()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.loss = torch.nn.BCELoss()
        self.model = NetworkRetriever.get_network(self.config_dict["network_name"])
        self.optimizer = optimizer_dict.get(self.config_dict["optimizer"])
        if self.optimizer is None:
            raise AssertionError(f'Optimizer must be one of the following: {list(optimizer_dict.keys())}')
        self.optimizer = self.optimizer(self.model.parameters(), lr=float(self.config_dict["learning_rate"]))

    def run_trials(self):
        self.results = {}
        for trial in range(self.total_trials):
            self.results[trial] = self.train(trial)
            print("-------------------")
            print(f"Best train accuracy for Trial {trial} is {max(self.results[trial]['train_acc'])}")
            print(f"Best val accuracy for Trial {trial} is {max(self.results[trial]['val_acc'])}")
            #print(f"Final test accuracy for Trial {trial} is {self.results[trial]['test_acc']}")
            print("-------------------")

    def train(self, trial):
        true_start = time.time()
        results_dict = dict(epoch_num=[], epoch_times=[], train_loss=[], train_acc=[], val_loss=[], val_acc=[], predicted=[], targets=[])
        for epoch in range(self.epochs):
            start_time = time.time()
            train_loss, train_accuracy, val_loss, val_accuracy, predicted, targets = \
                self.epoch_iteration()

            end_time = time.time()
            print(
                f"Trial {trial}/{self.total_trials - 1} | " +
                f"Epoch {epoch}/{self.epochs} Ended | " +
                "Total Time: {:.3f}s | ".format(end_time - true_start) +
                "Epoch Time: {:.3f}s | ".format(end_time - start_time) +
                "~Time Left: {:.3f}s | ".format(
                    (end_time - start_time) * (self.epochs - epoch + 1)),
                "Train Loss: {:.4f} | Train Acc. {:.4f}% | ".format(
                    train_loss,
                    train_accuracy) +
                "Val Loss: {:.4f} | Val Acc. {:.4f}%".format(val_loss,
                                                              val_accuracy))
            results_dict["epoch_num"].append(epoch)
            results_dict["epoch_times"].append(round(end_time - start_time, 3))
            results_dict["train_loss"].append(round(train_loss, 4))
            results_dict["val_loss"].append(round(val_loss, 4))
            results_dict["train_acc"].append(round(train_accuracy, 4))
            results_dict["val_acc"].append(round(val_accuracy, 4))
            results_dict["predicted"].append(predicted)
            results_dict["targets"].append(targets)


        # test_loss, test_accuracy = self.evaluate(self.val_loader, self.device)
        # results_dict["test_loss"] = test_loss
        # results_dict["test_acc"] = test_accuracy
        self.results_dict = results_dict
        return results_dict

    def evaluate(self, eval_loader, device):
        self.model.eval()
        eval_loss = 0
        correct = 0
        total = 0
        predicted_list = []
        targets_list = []
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(eval_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = self.model(inputs.float())
                loss = self.loss(outputs, targets.unsqueeze(1).float())

                eval_loss += loss.item()
                # _, predicted = outputs.max(1)
                predicted = torch.where(outputs > 0.5, 1, 0).squeeze()
                correct += np.where(targets == predicted, 1, 0).sum()
                total += targets.size(0)
                predicted_list += predicted.tolist()
                targets_list += targets.tolist()
        acc = 100. * correct / total
        return eval_loss / (batch_idx + 1), acc, predicted_list, targets_list

    def epoch_iteration(self):
        # breakpoint()
        self.model.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs.float())
            loss = self.loss(outputs, targets.unsqueeze(1).float())
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            train_loss += loss.item()
            predicted = torch.where(outputs > 0.5, 1, 0).squeeze()
            correct += np.where(targets==predicted, 1, 0).sum()
            total += targets.size(0)


        val_loss, val_accuracy, predicted_list, target_list = self.evaluate(self.val_loader, self.device)

        return (train_loss / (len(self.train_loader) + 1), 100. * correct / total,
                val_loss, val_accuracy, predicted_list, target_list)