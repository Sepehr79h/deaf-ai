class DataCreator:
    def __init__(self, config_dict):
        self.config_dict = config_dict

    def create_loaders(self):
        train_loader = None
        val_loader = None
        test_loader = None
        return train_loader, val_loader, test_loader
