
from torch.utils.data import Dataset, DataLoader
import sys
import torch
from PIL import Image
import os
import matplotlib.pyplot as plt

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
from data.preprocessing import get_transformation
class CustomDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

def create_paths_and_labels():
    prefixes = [
    "tshirt",
    "trousers",
    "pullover",
    "dress",
    "coat",
    "sandal",
    "shirt",
    "sneaker",
    "bag",
    "ankleboot"
    ]

    final_paths = []
    for prefix in prefixes:
        for i in range(0, 10):
            final_paths.append(R"../data/Real_world_testset/" + prefix + str(i) + ".jpg")
    labels = [label for label in range(10) for _ in range(10)]
    return final_paths, labels

def create_data_loader():
    transformation = get_transformation()
    image_paths, labels = create_paths_and_labels()
    test_dataset = CustomDataset(image_paths, labels, transform=transformation)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    return test_loader


