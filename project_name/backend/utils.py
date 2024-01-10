import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import io
import json

#from PIL import Image
#from ../models/prediction_model import ConvNet (we think this is the right way to import)
#from ../data/preprocessing import preprocessing
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#model = ConvNet(device).to(device)
# Load pre-trained weights
#model_weights_path = r"project_name\models\modelweights\model_weights.pth"
#model.load_state_dict(torch.load(model_weights_path, map_location=device))
# we dont know how you get the image but if you have an image called "ïmage" the next steps will work with that.
#Image.open(image_path)
#transformation = get_transformation()
#test_image = transformation(image)

#with torch.no_grad():
#    model.eval()
#    test_image = test_image.unsqueeze(0)
#    output = model(test_image)

# Get the predicted class index
#predicted_class = output.argmax().item()
#result = test_image[0, :,: ]

#now predicted_class is the prediction made by the model

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