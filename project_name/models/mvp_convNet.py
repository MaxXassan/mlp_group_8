import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from model import ConvNet

if __name__ == '__main__':
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Hyper-parameters
    num_epochs = 10
    batch_size = 64
    learning_rate = 0.001

    # Data transformation
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # Loading the FashionMNIST dataset
    train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True,
                                                    download=True, transform=transform)

    test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False,
                                                    download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = ConvNet(train_loader, test_loader, device, num_epochs, learning_rate).to(device)

    model.train_model()
    
    model.plots()
