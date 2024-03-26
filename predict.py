import numpy as np
import pandas as pd
import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from model import *
import torch.optim as optim
import torch.utils.data as Data

# use parser for implementation 
import argparse

def create_data_for_two_teams(pd_frame, home, away):
    team_dic = {}

    for idx in range(1521 ,len(pd_frame)):
        current_Homename = pd_frame.iloc[idx]['HomeTeam']
        current_row_home = pd.DataFrame(pd_frame.iloc[idx][:]).transpose()  

        if current_Homename in team_dic:
            team_dic[current_Homename] = pd.concat([team_dic[current_Homename], current_row_home], ignore_index=True)
        elif current_Homename == home:
            team_dic[current_Homename] = current_row_home

        current_awayname = pd_frame.iloc[idx]['AwayTeam']
        current_row_away = pd.DataFrame(pd_frame.iloc[idx][:]).transpose()  

        if current_awayname in team_dic:
            team_dic[current_awayname] = pd.concat([team_dic[current_awayname], current_row_away], ignore_index=True)
        elif current_awayname == away:
            team_dic[current_awayname] = current_row_away

    return team_dic


def create_data_for_each_team(pd_frame):
    team_dic = {}

    for idx in range(1521 ,len(pd_frame)):
        current_Homename = pd_frame.iloc[idx]['HomeTeam']
        current_row_home = pd.DataFrame(pd_frame.iloc[idx][:]).transpose()  

        if current_Homename in team_dic:
            team_dic[current_Homename] = pd.concat([team_dic[current_Homename], current_row_home], ignore_index=True)
        else:
            team_dic[current_Homename] = current_row_home

        current_awayname = pd_frame.iloc[idx]['AwayTeam']
        current_row_away = pd.DataFrame(pd_frame.iloc[idx][:]).transpose()  

        if current_awayname in team_dic:
            team_dic[current_awayname] = pd.concat([team_dic[current_awayname], current_row_away], ignore_index=True)
        else:
            team_dic[current_awayname] = current_row_away

    return team_dic

def rank_team_data(data1, data2):
    # Combine the dataframes
    combined_data = pd.concat([data1, data2], ignore_index=True)
    
    try:
        combined_data.iloc[:, 0] = pd.to_datetime(combined_data.iloc[:, 0], format='%Y-%m-%d')
    except:
        # Convert the first column to datetime format
        combined_data.iloc[:, 0] = pd.to_datetime(combined_data.iloc[:, 0], format='%Y/%m/%d')

    
    # Sort the dataframe by the first column (dates)
    combined_data.sort_values(by=combined_data.columns[0], inplace=True)
    
    # Extract columns from the 41st onward
    # Note: In Python, indexing starts at 0, so the 41st column is indexed as 40
    extracted_data = combined_data.iloc[:, 40:]

    return extracted_data
    

        
def create_dataset(data_dic, lookback_seqlen):
    X, y = [],[]
    # for team in data_dic:
    #     print(team)
    #     team_data = data_dic[team]
    #     for idx in range(len(team_data)-lookback_seqlen-1):
    #         data = team_data.iloc[idx:idx+lookback_seqlen, :-3].values
    #         labels_onehot = team_data.iloc[idx+lookback_seqlen+1, -3:].values
    #         X.append(data)
    #         y.append(labels_onehot)
    #         # data = pd.DataFrame(data)
    #         # labels_onehot = pd.DataFrame(labels_onehot)
    #         # X = pd.concat([X,data])
    #         # y = pd.concat([y,labels_onehot])

    for team1 in data_dic:
        for team2 in data_dic:
            if team1 == team2:
                pass
            else:
                # print("{} VS {}".format(team1, team2))
            
                team_data1 = data_dic[team1]
                team_data2 = data_dic[team2]
                team_data = rank_team_data(team_data1, team_data2)
                for idx in range(len(team_data)-lookback_seqlen-1):
                    data = team_data.iloc[idx:idx+lookback_seqlen, :-3].values
                    # print(data.shape)
                    # labels_onehot = team_data.iloc[idx+lookback_seqlen+1, -3:].values
                    # print(labels_onehot.shape)
                    X.append(data)
                    # y.append(labels_onehot)
                    # data = pd.DataFrame(data)
                    # labels_onehot = pd.DataFrame(labels_onehot)
                    # X = pd.concat([X,data])
                    # y = pd.concat([y,labels_onehot])

    # labels = np.argmax(np.stack(y), axis=1)
    # x_np = np.array(X)
    return torch.FloatTensor(X)


def get_df(args):
    dataframe = pd.read_csv(args.predict_data)
    print(dataframe.shape)
    return dataframe

def get_df_server():
    dataframe = pd.read_csv("Datasets/test_set_en.csv")
    return dataframe

def get_args():
    parser = argparse.ArgumentParser(description="parser for SNS coursework -- prediction")
    
    
    parser.add_argument('--predict_data', type=str, required=True, help='Path to the data for prediction')
    parser.add_argument('--load_model_path', type=str, required=True, help='Path to load the trained model for prediction')
    # Other settings
    parser.add_argument('--use_cuda', action='store_true', help='Use CUDA if available')
    parser.add_argument('--seq_len_required', type=int, required=True, help='The number of matches that need to look back')
    
    args = parser.parse_args()
    return args






def predict(device, team_series_data, model):

    # test_dataset = CSVDataset(csv_file_data=args.test_data, csv_file_label=args.test_label)
    # test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=True)
    model.to(device)
    model.eval()
    predicted_dataset = Data.TensorDataset(team_series_data)

    predicted_loader = DataLoader(
        dataset=predicted_dataset,
        batch_size=1,
        shuffle=False
    )



    with torch.no_grad():
        for data in predicted_loader:
            data = data[0]
            # print(data)
            data = data.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)

    # test_loss /= len(predicted_loader.dataset)
    # accuracy = 100. * correct / len(predicted_loader.dataset)
    return predicted


def load_model(path):
    model_loaded = torch.load(path)
    return model_loaded



def predict_server(HomeTeam, AwayTeam, seq_len):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FootballMatchPredictor(input_dim = 28, hidden_dim = 16, output_size=3)

    model_dic = load_model("path_to_save_model.pth")

    model.load_state_dict(model_dic)

    total_pre_data = get_df_server()
  
    data_dic_pre = create_data_for_two_teams(total_pre_data, HomeTeam, AwayTeam)

    team_series_pre_data = create_dataset(data_dic_pre, seq_len)

    result = predict(device, team_series_pre_data, model)

    return result.tolist()


if __name__ == "__main__":

    args = get_args()

    device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")

    model = FootballMatchPredictor(input_dim = 28, hidden_dim = 16, output_size=3)

    model_dic = load_model(args.load_model_path)

    model.load_state_dict(model_dic)

    total_pre_data = get_df(args)
  
    data_dic_pre = create_data_for_each_team(total_pre_data)

    team_series_pre_data = create_dataset(data_dic_pre, args.seq_len_required)

    result = predict(device, team_series_pre_data, model)

    print(result.tolist())




