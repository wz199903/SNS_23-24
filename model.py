import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F



# the lstm model for prediction
class RNNModel(nn.Module):
    def __init__(self, input_dim, units, output_size):
        super(RNNModel, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=units, batch_first=True)
        self.batch_norm = nn.BatchNorm1d(units)
        self.fc = nn.Linear(units, output_size)
        
    def forward(self, x):
        x = x.unsqueeze(-1) 
        # print(x.shape)
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Batch normalization
        # Need to reshape data for BatchNorm1d
        lstm_out = lstm_out[:, -1, :] 
        norm_out = self.batch_norm(lstm_out)
        
        # Dense layer to produce output
        out = self.fc(norm_out)
        return out
    

# A fully connect network as reference 
class MatchPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MatchPredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  
        self.fc2 = nn.Linear(hidden_size, hidden_size)  
        self.fc3 = nn.Linear(hidden_size, output_size)  
        
    def forward(self, x):
        x = F.relu(self.fc1(x))  
        x = F.relu(self.fc2(x)) 
        x = self.fc3(x)  
        return x
