import numpy as np
import pandas as pd
import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from model import *
import torch.optim as optim

# use parser for implementation 
import argparse

class CSVDataset(Dataset):
    def __init__(self, csv_file_data, csv_file_label):
        self.data_frame_data = pd.read_csv(csv_file_data)
        self.data_frame_label = pd.read_csv(csv_file_label)

    def __len__(self):
        return len(self.data_frame_data)
    
    def __getitem__(self, idx):
        x = self.data_frame_data.iloc[idx, 41:-3] 
        y = self.data_frame_label.iloc[idx]  
        

        # Convert to PyTorch tensors
        x = torch.tensor(x, dtype=torch.float32)
        # print(x.shape)


        # Convert one-hot encoded labels to class index
        y = torch.tensor(y, dtype=torch.float32)
        label_index = torch.argmax(y)  
        
        return x, label_index



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
    
    # Other settings
    parser.add_argument('--use_cuda', action='store_true', help='Use CUDA if available')
    
    args = parser.parse_args()
    return args



def train(args, model, device):

    train_dataset = CSVDataset(csv_file_data=args.train_data, csv_file_label=args.train_label)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
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
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
    
    torch.save(model.state_dict(), args.model_save_path)
    print('Training Complete.')


def test(args, model, device):

    test_dataset = CSVDataset(csv_file_data=args.test_data, csv_file_label=args.test_label)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=True)


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
    device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")

    # model = RNNModel(input_dim=1, units=1, output_size=3)
    model = MatchPredictor(27,2,3)

    train(args, model, device)

    test(args, model, device)



