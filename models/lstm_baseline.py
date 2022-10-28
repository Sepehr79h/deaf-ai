# import PyTorch
import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.lstm_1 = nn.LSTM(input_size=30, hidden_size=128, bidirectional=False, batch_first=True)
        self.lstm_2 = nn.LSTM(input_size=128, hidden_size=64, bidirectional=False, batch_first=True)
        self.lstm_2 = nn.LSTM(input_size=128, hidden_size=64, bidirectional=False, batch_first=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()

    def forward(self, input):
        output, _ = self.lstm_1(input)
        output = self.relu(output)
        output, _ = self.lstm_2(output)
        output = self.relu(output)
        _, (output, c_n) = self.lstm_3(output)
        output = self.relu(output)
        output = self.relu(self.linear1(output))
        output = self.relu(self.linear2(output))
        output = self.sigmoid(self.linear3(output))
        return output
