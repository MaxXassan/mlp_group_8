import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from torchvision import transforms
import numpy as np

# custom class that inherits from pytorch.DataSet. We reshape the data to 28x28 and transform it to a range
# be in the range [0, 1]
class CustomFashionMNIST(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_row = self.data.iloc[idx, 1:]  # Assuming first column is label
        image = np.array(image_row).astype('uint8').reshape(28, 28)  # Reshape to 28x28 for FashionMNIST
        label = self.data.iloc[idx, 0]  # First column is the label

        if self.transform:
            image = self.transform(image)

        return image, label

# transforms data to the range [0, 1]
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Custom dataset
train_dataset = CustomFashionMNIST(csv_file='./data/train.csv', transform=transform)
test_dataset = CustomFashionMNIST(csv_file='./data/test.csv', transform=transform)

# Data loaders
batch_size = 64  # Define your batch size
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)