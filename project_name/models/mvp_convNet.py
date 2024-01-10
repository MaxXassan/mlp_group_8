import torch
import torchvision
import torchvision.transforms as transforms

from model import ConvNet

if __name__ == '__main__':
    
    #use gpu is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #this batch size and learning rate showed best results
    num_epochs = 10
    batch_size = 100
    learning_rate = 0.002

    #normalize the data to range [0,1]
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    #loading the FashionMNIST dataset
    train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True,
                                                    download=True, transform=transform)

    test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False,
                                                    download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = ConvNet(train_loader, test_loader, device, num_epochs, learning_rate).to(device)

    model.train_model()
    
    model.plots()
