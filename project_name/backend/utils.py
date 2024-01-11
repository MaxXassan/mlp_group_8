import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import torch
import io
import json
import sys

from models.prediction_model import ConvNet
from data.preprocessing import get_transformation

device = 'cpu' #torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ConvNet().to(device)

model_weights_path = r"models\modelweights\model_weights.pth"
model.load_state_dict(torch.load(model_weights_path, map_location=device))

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

def transform_image(image_bytes):

    image = Image.open(io.BytesIO(image_bytes))

    transoformed_image = get_transformation()(image)

    return transoformed_image.unsqueeze(dim=0)

def get_prediction(transformed_image):
    model.eval()
    with torch.no_grad():
        y_hat = model(transformed_image).argmax(dim=1)
    return classes[y_hat.item()]

if __name__ == '__main__':
    print(sys.path[1])