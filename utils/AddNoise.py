import torch
import torchvision
import torchvision.transforms as transforms


# Define a custom transform to add noise
class AddNoiseTransform:
    def __init__(self, noise_level=0.1):
        """
        Initialize the noise transform.
        Args:
            noise_level (float): Standard deviation of the Gaussian noise to be added.
        """
        self.noise_level = noise_level

    def __call__(self, img):
        """
        Apply the transform to an image.
        Args:
            img (Tensor): Input image tensor.
        Returns:
            Tensor: Image tensor with added noise.
        """
        noise = torch.randn_like(img) * self.noise_level
        noisy_img = img + noise
        return noisy_img
