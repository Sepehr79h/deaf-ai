from matplotlib import pyplot as plt

class PlotCreator:
    def __init__(self, config_dict, results_dict):
        self.config_dict = config_dict
        self.results_dict = results_dict
        self.epochs = self.results_dict[0]["epoch_num"]
        self.train_loss_list = self.results_dict[0]["train_loss"]
        self.train_acc_list = self.results_dict[0]["train_acc"]
        self.val_loss_list = self.results_dict[0]["val_loss"]
        self.val_acc_list = self.results_dict[0]["val_acc"]

    def create_train_val_plot(self, x, y1, y2, plot_dict, metric_name="Accuracy"):
        plt.plot(x, y1, label=f"Training {metric_name}")
        plt.plot(x, y2, label=f"Validation {metric_name}")
        plt.title(plot_dict["title"])
        plt.xlabel(plot_dict["xlabel"])
        plt.ylabel(plot_dict["ylabel"])
        plt.legend()
        plt.savefig(f"{plot_dict['title']}.png", dpi=300, bbox_inches="tight", pad_inches=0.1)

        breakpoint()
        plt.clf()

    def generate_plots(self):
        acc_plot = {
            "title": f"Accuracy vs Epochs {self.config_dict['network_name'], self.config_dict['learning_rate']}",
            "xlabel": "Epoch",
            "ylabel": "Accuracy",
        }
        loss_plot = {
            "title": f"Loss vs Epochs {self.config_dict['network_name'], self.config_dict['learning_rate']}",
            "xlabel": "Epoch",
            "ylabel": "Loss",
        }
        self.create_train_val_plot(self.epochs, self.train_acc_list, self.val_acc_list, acc_plot)
        self.create_train_val_plot(self.epochs, self.train_loss_list, self.val_loss_list, loss_plot)







