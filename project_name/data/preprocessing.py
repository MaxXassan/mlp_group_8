import torchvision.transforms as transforms
# Data transformation
#transform = transforms.Compose([
#    transforms.ToTensor()
#])
# Apply the same transform used for training

def get_transformation():
    transformation = transforms.Compose([
        transforms.Resize((28,28)),
        transforms.RandomInvert(p=1),
        transforms.Grayscale(),
        transforms.ToTensor()
    ])
    return transformation