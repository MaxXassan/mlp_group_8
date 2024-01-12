import torchvision.transforms as transforms

#preprocessing pipeline.
#the images are resized to 28x28, then inverted and converted to greyscale.
def get_transformation():
    transformation = transforms.Compose([
        transforms.Resize((28,28)),
        transforms.RandomInvert(p=1),
        transforms.Grayscale(),
        transforms.ToTensor()
    ])
    return transformation