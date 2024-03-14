import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms




class RNNModel(nn.Module):
    def __init__(self, input_dim, units, output_size):
        super(RNNModel, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=units, batch_first=True)
        self.batch_norm = nn.BatchNorm1d(units)
        self.fc = nn.Linear(units, output_size)
        
    def forward(self, x):
        # LSTM layer
        # x should be of shape (batch, seq_len, input_dim), seq_len is assumed to be 1 for each feature vector of dimension input_dim
        # Output shape of lstm_out is (batch, seq_len, hidden_size)
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Batch normalization
        # Need to reshape data for BatchNorm1d
        lstm_out = lstm_out[:, -1, :] # Reshape to keep only the last sequence output for batch norm
        norm_out = self.batch_norm(lstm_out)
        
        # Dense layer to produce output
        out = self.fc(norm_out)
        return out
