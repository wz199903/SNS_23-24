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


# def create_data_for_each_team(pd_frame):
    
#     # team_dic = {}
#     # for idx, _ in enumerate(pd_frame):
#     #     current_Homename = pd_frame.iloc[idx]['HomeTeam']
#     #     if current_Homename in team_dic:
#     #         # team_dic[current_Homename].append(pd_frame.iloc[idx][40:])
#     #         # team_dic[current_Homename] = np.append(team_dic[current_Homename], pd_frame.iloc[idx][41:])
#     #         team_dic[current_Homename] = pd.concat([team_dic[current_Homename], pd_frame.iloc[idx][40:]])
#     #     else:
#     #         team_dic[current_Homename] = pd.DataFrame()
#     #         team_dic[current_Homename] = pd.concat([team_dic[current_Homename], pd_frame.iloc[idx][40:]])

#     #     current_awayname = pd_frame.iloc[idx]['AwayTeam']
#     #     if current_awayname in team_dic:
#     #         team_dic[current_awayname] = pd.concat([team_dic[current_awayname], pd_frame.iloc[idx][40:]])
#     #         # team_dic[current_awayname] = np.append(team_dic[current_awayname], pd_frame.iloc[idx][41:])
#     #     else:
#     #         team_dic[current_awayname] = pd.DataFrame()
#     #         team_dic[current_awayname] = pd.concat([team_dic[current_awayname], pd_frame.iloc[idx][40:]])

#     # return team_dic

def create_data_for_each_team(pd_frame):
    team_dic = {}

    for idx in range(len(pd_frame)):
        current_Homename = pd_frame.iloc[idx]['HomeTeam']
        current_row_home = pd.DataFrame(pd_frame.iloc[idx][40:]).transpose()  # Convert to DataFrame and transpose

        if current_Homename in team_dic:
            team_dic[current_Homename] = pd.concat([team_dic[current_Homename], current_row_home], ignore_index=True)
        else:
            team_dic[current_Homename] = current_row_home

        current_awayname = pd_frame.iloc[idx]['AwayTeam']
        current_row_away = pd.DataFrame(pd_frame.iloc[idx][40:]).transpose()  # Convert to DataFrame and transpose

        if current_awayname in team_dic:
            team_dic[current_awayname] = pd.concat([team_dic[current_awayname], current_row_away], ignore_index=True)
        else:
            team_dic[current_awayname] = current_row_away

    # Optionally, you can reset index here if needed, but ignore_index in pd.concat() should handle it

    return team_dic


        
def create_dataset(data_dic, lookback_seqlen):
    X, y = [],[]
    for team in data_dic:
        print(team)
        team_data = data_dic[team]
        for idx in range(len(team_data)-lookback_seqlen-1):
            data = team_data.iloc[idx:idx+lookback_seqlen, :-3].values
            # print(data.shape)
            labels_onehot = team_data.iloc[idx+lookback_seqlen+1, -3:].values
            # print(labels_onehot.shape)
            X.append(data)
            y.append(labels_onehot)
            # data = pd.DataFrame(data)
            # labels_onehot = pd.DataFrame(labels_onehot)
            # X = pd.concat([X,data])
            # y = pd.concat([y,labels_onehot])

    labels = np.argmax(np.stack(y), axis=1)  # Adjust axis if necessary based on your labels_onehot structure
    return torch.tensor(X, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)   

# class CSVDataset(Dataset):
#     def __init__(self, csv_file_data, csv_file_label, seq_length):
#         self.data_frame_data = pd.read_csv(csv_file_data),seq_length
#         self.data_frame_label = pd.read_csv(csv_file_label),seq_length
#         self.seq_length = seq_length


#     def __len__(self):
#         return len(self.data_frame_data) - self.seq_length
    
#     def __getitem__(self, idx):
#         x = self.data_frame_data.iloc[idx:idx+self.seq_length, 26:-3].values
#         y = self.data_frame_label.iloc[idx+self.seq_length-1, 26:-3].values  
        
#         x = torch.tensor(x, dtype=torch.float32)
#         y = torch.tensor(y, dtype=torch.float32)
#         label_index = torch.argmax(y)

#         return x, label_index

def get_df(args):
    dataframe = pd.read_csv(args.train_data)
    print(dataframe.shape)
    return dataframe

def get_test_df(args):
    dataframe = pd.read_csv(args.test_data)
    return dataframe

def get_args():
    parser = argparse.ArgumentParser(description="parser for SNS coursework model training and testing")
    
    # Model hyperparameters
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    
    # Train and test data set
    parser.add_argument('--train_data', type=str, required=True, help='Path to the training data')
    parser.add_argument('--test_data', type=str, required=True, help='Path to the test data')

    # Label
    parser.add_argument('--train_label', type=str, required=True, help='Path to the training data')
    parser.add_argument('--test_label', type=str, required=True, help='Path to the test data')
    
    # Model settings
    parser.add_argument('--model_save_path', type=str, default='./model.pth', help='Where to save the trained model')
    parser.add_argument('--seq_len', type=int, default=10, help='seq_len of recurrent model')
    
    # Other settings
    parser.add_argument('--use_cuda', action='store_true', help='Use CUDA if available')
    
    args = parser.parse_args()
    return args




def train(args, model, device, team_series_data, team_series_label):
    seq_length = args.seq_len  # Define the sequence length based command line parameters
    train_dataset = Data.TensorDataset(team_series_data, team_series_label)

    print(team_series_data.shape)
    print(team_series_label.shape)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )

    # train_dataset = CSVDataset(csv_file_data=args.train_data, csv_file_label=args.train_label)
    # train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)


    # # Load your dataset
    # train_dataset = CSVDataset(csv_file_data=args.train_data, csv_file_label='your_label_file.csv')  # Adjust the label file argument accordingly
    # train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    
    # Define model, loss, and optimizer
    model.to(device)
    criterion = nn.CrossEntropyLoss() 
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Training loop
    for epoch in range(args.epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            # target = target.unsqueeze(-1)
            # target = target[:, -1]
            data, target = data.to(device), target.to(device)
            # print(data)
            # print(data.shape)                   
            # print(target)
            # target = target.squeeze(1)
            # print(target.shape)
            optimizer.zero_grad()
            output = model(data)
            # print(output.shape)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
    
    torch.save(model.state_dict(), args.model_save_path)
    print('Training Complete.')


def test(args, model, device, team_series_data, team_series_label):

    # test_dataset = CSVDataset(csv_file_data=args.test_data, csv_file_label=args.test_label)
    # test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=True)

    test_dataset = Data.TensorDataset(team_series_data, team_series_label)

    print(team_series_data.shape)
    print(team_series_label.shape)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )


    criterion = nn.CrossEntropyLoss()  
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    

    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            correct += (predicted == target).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.0f}%)')






if __name__ == "__main__":


    args = get_args()

    total_data = get_df(args)

    data_dic = create_data_for_each_team(total_data)

    team_series_data, team_series_label = create_dataset(data_dic, args.seq_len)


    device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")

    model = FootballMatchPredictor(input_dim = 28, hidden_dim=16, output_size=3)
    # model = MatchPredictor(42,2,3)

    train(args, model, device, team_series_data, team_series_label)

    # test(args, model, device, team_series_data)

    total_test_data = get_test_df(args)

    data_dic_test = create_data_for_each_team(total_test_data)

    team_series_test_data, team_series_test_label = create_dataset(data_dic_test, args.seq_len)

    test(args, model, device, team_series_test_data, team_series_test_label)


