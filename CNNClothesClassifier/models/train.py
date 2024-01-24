import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split

from model import ConvNet
from base_line import BaseLine

def train_CNN_model(train_loader, eval_loader, device, num_epochs, learning_rate, weight_decay):
    # Initializing the model
    model = ConvNet(train_loader, eval_loader, device, num_epochs, learning_rate, weight_decay).to(device)
    
    # Trains, evaluates, and saves the model
    model.train_model()

    # Plotting the model
    model.plots()

def train_baseline_model(train_loader, eval_loader, device, num_epochs, learning_rate):
    # Initializing the model
    model = BaseLine(train_loader, eval_loader, device, num_epochs, learning_rate).to(device)
    
    # Trains, evaluates, and saves the model
    model.train_model()

    # Plotting the model
    model.plots()

if __name__ == '__main__':
    # Device configuration
    device = 'cpu'
    num_epochs = 10
    batch_size = 4
    learning_rate = 0.001
    weight_decay=0.0005

    # Data transformation
    transformation = transforms.Compose([
        transforms.ToTensor()
    ])

    # Loading the FashionMNIST dataset
    train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True,
                                                    download=True, transform=transformation)


    
    #split training data into 80/20 training/evaluation set, test set is only used for evaluation once training is done.
    train_size = int(len(train_dataset) * 0.8) 
    eval_size = len(train_dataset) - train_size

    train_subset, eval_subset = random_split(train_dataset, [train_size, eval_size])

    eval_loader =  torch.utils.data.DataLoader(eval_subset, batch_size=batch_size, shuffle=True)
    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size, shuffle=True)

    '''Choose which model you want to train
    '''
    train_CNN_model(train_loader, eval_loader, device, num_epochs, learning_rate, weight_decay)
    # train_baseline_model(train_loader, eval_loader, device, num_epochs, learning_rate)