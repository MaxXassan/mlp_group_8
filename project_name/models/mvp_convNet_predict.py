import torch
import torchvision
import torchvision.transforms as transforms

from prediction_model import ConvNet
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def load_image(image_path):
    # Load the image using PIL
    image = Image.open(image_path) # Convert to grayscale if needed

    return image

def show_image_with_prediction(image, predicted_class):
    # Display the image using matplotlib
    plt.imshow(np.array(image))
    plt.title(f'Predicted Class: {predicted_class}')
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



    # Data transformation
    #transform = transforms.Compose([
    #    transforms.ToTensor()
    #])
    # Apply the same transform used for training
    transformation = transforms.Compose([
        transforms.Resize((28,28)),
        transforms.RandomInvert(p=1),
        transforms.Grayscale(),
        transforms.ToTensor()
    ])

 

    # Initialize your model
    model = ConvNet(device).to(device)


    # Load pre-trained weights
    model_weights_path = r"project_name\models\modelweights\model_weights.pth"
    model.load_state_dict(torch.load(model_weights_path, map_location=device))

    # Apply the transform and add a batch dimension
    #image = transform(image).unsqueeze(0)
    # Test the model on a single image
    image_path = r"C:\Users\jelle\Desktop\mlp\figures\1836050e6a1b47df90a8bd400b92a5c5.jpg"
    test_image = load_image(image_path)

    test_image = transformation(test_image)


    test_image_tensor = test_image
    #test_image.show()
    print("format: ", test_image_tensor.shape)

    with torch.no_grad():
        model.eval()
        test_image_tensor = test_image_tensor.unsqueeze(0)
        output = model(test_image_tensor)

    # Get the predicted class index
    predicted_class = output.argmax().item()
    result = test_image[0, :,: ]
    show_image_with_prediction(result, predicted_class)
    print(f'The predicted class index for the test image is: {predicted_class}')