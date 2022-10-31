from models.lstm_baseline import BaselineModel
from models.mobilenet import MobileNet


class NetworkRetriever:
    @staticmethod
    def get_network(network_name):
        model_dict = dict(BaselineModel=BaselineModel(), MobileNet=MobileNet())
        model = model_dict.get(network_name)
        if model is None:
            raise AssertionError(f"Invalid model name, must be one of the following: {list(model_dict.keys())}")
        return model
