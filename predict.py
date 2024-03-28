import numpy as np
import pandas as pd
import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader , TensorDataset
from torchvision import datasets, transforms
from model import *
import torch.optim as optim
import torch.utils.data as Data
from datetime import datetime

hid_dim = 32
epochs = 50 

def create_dataset_predict(data_in, feature_selected, match_date, seq_len, found_team):
    # print(len(data_in))
    # print(1737 - seq_len)
    # print(match_date)
    date_obj = datetime.strptime(match_date, "%Y-%m-%d %H:%M:%S")
    new_date_str = f"{date_obj.year}/{date_obj.month}/{date_obj.day}"
    print(new_date_str)
    startrow = 1737 - seq_len - 1
    data_in = data_in.iloc[startrow:, :]
    df_selected  = data_in[feature_selected]
    date_col = data_in['Date']
    home_col = data_in['HomeTeam']
    away_col = data_in['AwayTeam']

    # labels_onehot = data_in.iloc[:, -3:].values
    for idx in range(len(data_in)):
        if new_date_str == date_col.iloc[idx] and (found_team[0] == home_col.iloc[idx] or found_team[0] == away_col.iloc[idx] or found_team[1] == home_col.iloc[idx] or found_team[1] == away_col.iloc[idx]):
            print("------idx------")
            print(idx)
            newdf = df_selected.iloc[idx - seq_len : idx, :]
            return torch.tensor(newdf.values, dtype=torch.float)
    
    return None
    # labels = np.argmax(np.stack(labels_onehot), axis=1)  
    
    

def predict_server(match_date, seq_len, found_team):
    dataframe = pd.read_csv("Datasets/test_set_en.csv")
    
    # team_series_data = np.reshape(team_series_data, (1803,1,len(feature_selected)))
    feature_selected = ['HTGS_norm', 'ATGS_norm', 'HTGC_norm', 'ATGC_norm',
                        'HTP_norm', 'ATP_norm', 'HM1', 'AM1', 'HM2', 'AM2',
                        'HM3', 'AM3', 'HM4', 'AM4', 'HM5', 'AM5', 'DiffLP', 'HomeTeamRk', 'AwayTeamRk']

    dataset = create_dataset_predict(dataframe, feature_selected, match_date, seq_len, found_team)
    if dataset == None:
        # TODO
        return "ERROR"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FootballMatchPredictor(input_dim = len(feature_selected), hidden_dim = hid_dim, output_size=3)

    model_dic = torch.load("model_saved/trained.pth")

    model.load_state_dict(model_dic)

    model.to(device)
    model.eval()
    print(dataset.shape)

    dataset_tensor  = np.reshape(dataset, (1, len(dataset), 19))
    
    predicted_dataset = TensorDataset(dataset_tensor)

    predicted_loader = DataLoader(
        dataset=predicted_dataset,
        batch_size=1,
        shuffle=False
    )
    print(predicted_loader)



    with torch.no_grad():
        for data in predicted_loader:
            data = data[0]
            print(data.shape)
            data = data.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            print("prediction: {}".format(predicted))
            
    # test_loss /= len(predicted_loader.dataset)
    # accuracy = 100. * correct / len(predicted_loader.dataset)
    return predicted


