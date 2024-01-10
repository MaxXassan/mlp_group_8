import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import io
import json

imagenet_class_index = json.load(open('imagenet_class_index.json'))

model = models.densenet121(weights='IMAGENET1K_V1')
model.eval()

def transform_image(image_bytes):
    transformer = transforms.Compose([transforms.Resize(255), 
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])

    image = Image.open(io.BytesIO(image_bytes))

    return transformer(image).unsqueeze(dim=0)

def get_prediction(transformed_image):
    y_hat = model(transformed_image).argmax(dim=1)
    return imagenet_class_index[str(y_hat.item())]