import torch
import torchvision
import torchvision.transforms as transforms

from model import ConvNet

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

    test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False,
                                                    download=True, transform=transformation)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initializing the model
    model = ConvNet(train_loader, test_loader, device, num_epochs, learning_rate,weight_decay).to(device)

    # Trains, evaluates, and saves the model
    model.train_model()

    # Plotting the model
    model.plots()
