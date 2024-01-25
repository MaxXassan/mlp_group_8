import torch
import torchvision
import torchvision.transforms as transforms
import os
import sys
from model import ConvNet
from base_line import BaseLine

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
from data.custom_dataset import create_data_loader

def test_CNN_model(test_loader, device, project_root):
    
    #Get the weights
    sys.path.append(project_root)
    cnn_weights_path = os.path.join(project_root, 'models', 'modelweights', 'model_weights.pth') 
    # Initializing the model
    model = ConvNet(None, test_loader, device, None, None, None).to(device)
    # Load the weights
    model.load_state_dict(torch.load(cnn_weights_path , map_location=device))
    
    # Tests the model, and plots the confusion matrix, f1, and accuracy
    model.test_model(test_loader)

def test_baseline_model(test_loader, device, project_root):

    #Get the weights
    sys.path.append(project_root)
    baseline_weights_path = os.path.join(project_root, 'models', 'modelweights', 'baseline_model_weights.pth') 
    # Initializing the model
    model = BaseLine(None, test_loader, device, None, None).to(device)
    # Load the weights
    model.load_state_dict(torch.load(baseline_weights_path, map_location=device))
    
    model.test_model(test_loader)

if __name__ == '__main__':
    #get the path of the script
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


    baseline_weights_path = os.path.join(project_root, 'models', 'modelweights', 'baseline_model_weights.pth') 

    device = 'cpu'

    transformation = transforms.Compose([
        transforms.ToTensor()
    ])

    test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False,
                                                download=True, transform=transformation)
    
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

    real_world_loader = create_data_loader()


    '''Choose which model you want to test
    '''
    #for testing on testdata
    test_CNN_model(test_loader, device, project_root)
    test_baseline_model(test_loader, device, project_root)

    #for testing on real world data
    # test_CNN_model(real_world_loader, device, project_root)
    # test_baseline_model(real_world_loader, device, project_root)


