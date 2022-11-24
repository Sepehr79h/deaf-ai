# import PyTorch
import torch
import torch.nn as nn

class PoseLSTM(nn.Module):
    def __init__(self):
        super(PoseLSTM, self).__init__()
        self.lstm1 = nn.LSTM(input_size=38, hidden_size=1024, bidirectional=False, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=1024, hidden_size=512, bidirectional=False, batch_first=True)
        self.lstm3 = nn.LSTM(input_size=512, hidden_size=256, bidirectional=False, batch_first=True)
        self.linear1 = nn.Linear(256, 128)
        self.linear2 = nn.Linear(128, 64)
        self.linear3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()

    def forward(self, input):
        output, _ = self.lstm1(input)
        output = self.relu(output)
        output, _ = self.lstm2(output)
        output = self.relu(output)
        _, (output, c_n) = self.lstm3(output)
        output = self.relu(output)
        output = self.relu(self.linear1(output))
        output = self.relu(self.linear2(output))
        output = self.sigmoid(self.linear3(output)).squeeze(0)
        return output




