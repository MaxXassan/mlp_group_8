import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

from model import ConvNet

if __name__ == '__main__':
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.append(project_root)
    model_weights_path = os.path.join(project_root, 'models', 'modelweights', 'model_weights.pth')  

    device = 'cpu'

    transformation = transforms.Compose([
        transforms.ToTensor()
    ])

    test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False,
                                                download=True, transform=transformation)
    
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = ConvNet(None, test_loader, device, None, None).to(device)

    model.load_state_dict(torch.load(model_weights_path, map_location=device))

    model.test_model(test_loader)

