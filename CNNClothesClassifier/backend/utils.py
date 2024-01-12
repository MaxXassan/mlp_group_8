import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import torch
import io
import json
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from models.model import ConvNet
from data.preprocessing import get_transformation
# only use cpu to ensure it works on all computers.
device = 'cpu'
#create the model and load in the weights.
model = ConvNet().to(device)
model_weights_path = os.path.join(project_root, 'models', 'modelweights', 'model_weights.pth')
model.load_state_dict(torch.load(model_weights_path, map_location=device))
#dictionary to get the correct classname for the predicted class.
classes = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot"
}
#preprocess the image.
def transform_image(image_bytes):
    """
    """
    image = Image.open(io.BytesIO(image_bytes))

    transoformed_image = get_transformation()(image)

    return transoformed_image.unsqueeze(dim=0)
#get the prediction and return it.
def get_prediction(transformed_image):
    model.eval()
    with torch.no_grad():
        y_hat = model(transformed_image).argmax(dim=1)
    return classes[y_hat.item()]