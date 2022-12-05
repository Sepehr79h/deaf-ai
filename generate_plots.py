from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np

class PlotCreator:
    def __init__(self, config_dict, results_dict):
        self.config_dict = config_dict
        self.results_dict = results_dict
        self.epochs = self.results_dict["epoch_num"]#[0]["epoch_num"]
        self.train_loss_list = self.results_dict["train_loss"]#[0]["train_loss"]
        self.train_acc_list = self.results_dict["train_acc"]#[0]["train_acc"]
        self.val_loss_list = self.results_dict["val_loss"]#[0]["val_loss"]
        self.val_acc_list = self.results_dict["val_acc"]#[0]["val_acc"]

    def create_train_val_plot(self, x, y1, y2, plot_dict, metric_name="Accuracy"):
        plt.plot(x, y1, label=f"Training {metric_name}")
        plt.plot(x, y2, label=f"Validation {metric_name}")
        plt.title(plot_dict["title"])
        plt.xlabel(plot_dict["xlabel"])
        plt.ylabel(plot_dict["ylabel"])
        plt.legend()
        plt.savefig(f"{plot_dict['title']}.png", dpi=300, bbox_inches="tight", pad_inches=0.1)
        plt.clf()

    def generate_plots(self):
        acc_plot = {
            "title": "Accuracy vs Epochs",
            "xlabel": "Epoch",
            "ylabel": "Accuracy",
            "file_name": f"Accuracy vs Epochs {self.config_dict['network_name'], self.config_dict['learning_rate']}"
        }
        loss_plot = {
            "title": "Loss vs Epochs",
            "xlabel": "Epoch",
            "ylabel": "Loss",
            "file_name": f"Loss vs Epochs {self.config_dict['network_name'], self.config_dict['learning_rate']}"
        }
        self.create_train_val_plot(self.epochs, self.train_acc_list, self.val_acc_list, acc_plot)
        self.create_train_val_plot(self.epochs, self.train_loss_list, self.val_loss_list, loss_plot)

    def create_confusion_matrix(self):
        label_conversion = {0: "Not Signing", 1: "Signing"}
        targets = [label_conversion[label] for label in self.results_dict["targets"][-1]]
        predictions = [label_conversion[label] for label in self.results_dict["predicted"][-1]]
        cm = confusion_matrix(targets, predictions)
        # plt.imshow(cm, cmap=plt.cm.Blues)
        plt.xlabel("Predicted labels")
        plt.ylabel("True labels")
        plt.title("Confusion Matrix")

        import seaborn as sns
        ax = sns.heatmap(cm/np.sum(cm), annot=True,
            fmt='.2%', cmap='Blues')
        ax.xaxis.set_ticklabels(['Not Signing', 'Signing'])
        ax.yaxis.set_ticklabels(['Not Signing', 'Signing'])

        #plt.colorbar()
        plt.savefig("Confusion Matrix.png", dpi=300, bbox_inches="tight", pad_inches=0.1)
        plt.clf()
        breakpoint()







