import torchvision.transforms as transforms


def get_transformation():
    transformation = transforms.Compose([
        transforms.Resize((28,28)),
        transforms.RandomInvert(p=1),
        transforms.Grayscale(),
        transforms.ToTensor()
    ])
    return transformation