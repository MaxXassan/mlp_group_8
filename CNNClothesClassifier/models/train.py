import torch
import torchvision
import torchvision.transforms as transforms

from model import ConvNet
from base_line import BaseLine

if __name__ == '__main__':
    # Device configuration
    device = 'cpu'

    num_epochs = 10
    batch_size = 128
    learning_rate = 0.001

    # Data transformation
    transformation = transforms.Compose([
        transforms.ToTensor()
    ])

    # Loading the FashionMNIST dataset
    train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True,
                                                    download=True, transform=transformation)

    test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False,
                                                    download=True, transform=transformation)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initializing the model
    # model = ConvNet(train_loader, test_loader, device, num_epochs, learning_rate).to(device)
    model = BaseLine(train_loader, test_loader, device, num_epochs, learning_rate).to(device)
    
    # Trains, evaluates, and saves the model
    model.train_model()

    # Plotting the model
    # model.plots()
