
from torch import nn
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

        tag_scores = F.softmax(tag_space, dim=1)
        # print("tag:{}".format(tag_scores))
        
        return tag_scores

