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


def create_data_for_each_team(pd_frame):
    team_dic = {}

    for idx in range(len(pd_frame)):
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
                    labels_onehot = team_data.iloc[idx+lookback_seqlen+1, -3:].values
                    # print(labels_onehot.shape)
                    X.append(data)
                    y.append(labels_onehot)
                    # data = pd.DataFrame(data)
                    # labels_onehot = pd.DataFrame(labels_onehot)
                    # X = pd.concat([X,data])
                    # y = pd.concat([y,labels_onehot])
    # x_np = np.array(X)
    # y_np = np.array(y)
    labels = np.argmax(np.stack(y), axis=1)  
    return torch.FloatTensor(X), torch.LongTensor(labels)   


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
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.00001, help='Learning rate')
    
    # Train and test data set
    parser.add_argument('--train_data', type=str, required=True, help='Path to the training data')
    parser.add_argument('--test_data', type=str, required=True, help='Path to the test data')

    # Label
    parser.add_argument('--train_label', type=str, required=True, help='Path to the training data')
    parser.add_argument('--test_label', type=str, required=True, help='Path to the test data')
    
    # Model settings
    parser.add_argument('--model_save_path', type=str, default='./model.pth', help='Where to save the trained model')
    parser.add_argument('--seq_len', type=int, default=5, help='seq_len of recurrent model')
    parser.add_argument('--hid_dim', type=int, default=16, help='hidden dimension of recurrent model')
    
    # Other settings
    parser.add_argument('--use_cuda', action='store_true', help='Use CUDA if available')
    
    args = parser.parse_args()
    return args




def train(args, model, device, team_series_data, team_series_label):
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
        correct = 0
        total = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            # print(output.shape)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            # if batch_idx % 100 == 0:
            #     print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item() 

            if batch_idx % 100 == 0:
                accuracy = 100. * correct / total
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                    f'({100. * batch_idx / len(train_loader):.0f}%)]\t'
                    f'Loss: {loss.item():.6f} Accuracy: {accuracy:.2f}%')
    torch.save(model.state_dict(), args.model_save_path)
    print('Training Complete.')


def test(args, model, device, team_series_data, team_series_label):

    # test_dataset = CSVDataset(csv_file_data=args.test_data, csv_file_label=args.test_label)
    # test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=True)
    model.eval()
    test_dataset = Data.TensorDataset(team_series_data, team_series_label)

    # print(team_series_data.shape)
    # print(team_series_label.shape)
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

    model = FootballMatchPredictor(input_dim = 28, hidden_dim = args.hid_dim, output_size=3)
    
    train(args, model, device, team_series_data, team_series_label)

    total_test_data = get_test_df(args)
  
    data_dic_test = create_data_for_each_team(total_test_data)

    team_series_test_data, team_series_test_label = create_dataset(data_dic_test, args.seq_len)

    test(args, model, device, team_series_test_data, team_series_test_label)


