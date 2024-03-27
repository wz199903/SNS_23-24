import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F



import torch
import torch.nn as nn
import torch.nn.functional as F

class FootballMatchPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_size):
        """
        input_dim: Number of features per time step in the input.
        hidden_dim: The number of features in the hidden state h of the LSTM.
        output_size: The number of classes for prediction (e.g., win, lose, draw).
        """
        super(FootballMatchPredictor, self).__init__()
        self.hidden_dim = hidden_dim

        # Unidirectional LSTM because of future prediction
        self.lstm = nn.LSTM(input_dim, hidden_dim, bidirectional=False)
        self.relu = nn.ReLU()
        # Mapping from LSTM output to the prediction space W/L/D
        self.hidden2tag = nn.Linear(hidden_dim, output_size)

    def forward(self, match_sequences):
        lstm_out, _ = self.lstm(match_sequences)
        
        last_timestep_output = lstm_out[:, -1, :]
        
        # Pass the output of the last timestep through the linear layer
        tag_space = self.hidden2tag(self.relu(last_timestep_output))

        tag_scores = F.log_softmax(tag_space, dim=1)
        # print("tag:{}".format(tag_scores))
        
        return tag_scores
    

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
