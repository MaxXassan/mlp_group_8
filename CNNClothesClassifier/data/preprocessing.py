import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from scipy import signal

#preprocessing pipeline.
#the images are resized to 28x28, then inverted and converted to greyscale.
def get_transformation():
    transformation = transforms.Compose([
        transforms.Resize((28,28)),
        transforms.RandomInvert(p=1),
        transforms.Grayscale(),
        GaussianSharpeningTransform(radius=1, std_dev=1, intensity=1),
        transforms.ToTensor()
    ])
    return transformation
"""
This function takes an image and applies Gaussian sharpening to it.
"""
class GaussianSharpeningTransform:
    def __init__(self, radius=1.0, std_dev=1.0, intensity=1.0):
        self.radius = radius
        self.std_dev = std_dev
        self.intensity = intensity

    def __call__(self, img):
        img_tensor = transforms.ToTensor()(img)

        kernel_size = int(self.radius) * 2 + 1
        kernel = self.create_gaussian_kernel(kernel_size, self.std_dev)

        blurred = F.conv2d(img_tensor.unsqueeze(0), kernel, padding=kernel_size//2)

        edge_mask = img_tensor - blurred.squeeze(0)

        sharpened = img_tensor + self.intensity * edge_mask

        sharpened = torch.clamp(sharpened, 0, 1)

        return transforms.ToPILImage()(sharpened)

    """
    This function generates the actual gaussian kernel that is used for blurring of the image.
    """
    def create_gaussian_kernel(self, kernel_size, std_dev):
        kernel = torch.from_numpy(np.outer(signal.gaussian(kernel_size, std_dev), signal.gaussian(kernel_size, std_dev)))
        kernel = kernel / torch.sum(kernel)
        return kernel.float().unsqueeze(0).unsqueeze(0)
