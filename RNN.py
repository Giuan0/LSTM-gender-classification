import torch.nn as nn
import torch

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_first=True):
        super(RNN, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=batch_first
        )
        self.linear = torch.nn.Linear(20, 1)
    
    def forward(self, x):
        x, (h_n, c_n) = self.lstm(x, None)
        x = nn.Sigmoid()(self.linear(x[:, -1, :])).view(1,1,1) #batches can contain one one sample, for now
        
        return x
