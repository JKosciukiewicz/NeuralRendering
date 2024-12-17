import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def plot_sample_noise(img_path):
    # Open the image
    sample_image = Image.open(img_path)

    # Convert the image to a NumPy array
    image_array = np.array(sample_image, dtype=np.float32)

    # Add Gaussian noise
    def add_gaussian_noise(image, mean=0, std=25):
        noise = np.random.normal(mean, std, image.shape)
        noisy_image = image + noise
        # Clip the values to be within valid range (0-255 for an image)
        noisy_image = np.clip(noisy_image, 0, 255)
        return noisy_image

    # Add Salt-and-Pepper noise
    def add_salt_and_pepper_noise(image, amount=0.05):
        noisy_image = image.copy()
        num_salt = int(amount * image.size * 0.5)
        num_pepper = int(amount * image.size * 0.5)

        # Add salt (white pixels)
        coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape]
        noisy_image[coords[0], coords[1], :] = 255

        # Add pepper (black pixels)
        coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape]
        noisy_image[coords[0], coords[1], :] = 0

        return noisy_image

    # Add Poisson noise
    def add_poisson_noise(image):
        noisy_image = np.random.poisson(image).astype(np.float32)
        noisy_image = np.clip(noisy_image, 0, 255)
        return noisy_image

    # Add Speckle noise
    def add_speckle_noise(image, std=0.2):
        noise = np.random.randn(*image.shape) * std
        noisy_image = image + image * noise
        noisy_image = np.clip(noisy_image, 0, 255)
        return noisy_image

    # Apply different types of noise
    noisy_gaussian = add_gaussian_noise(image_array)
    noisy_salt_pepper = add_salt_and_pepper_noise(image_array, amount=0.05)
    noisy_poisson = add_poisson_noise(image_array)
    noisy_speckle = add_speckle_noise(image_array)

    # Display the original and noisy images
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 3, 1)
    plt.title('Original Image')
    plt.imshow(sample_image)
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.title('Gaussian Noise')
    plt.imshow(Image.fromarray(noisy_gaussian.astype(np.uint8)))
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.title('Salt-and-Pepper Noise')
    plt.imshow(Image.fromarray(noisy_salt_pepper.astype(np.uint8)))
    plt.axis('off')

    plt.subplot(2, 3, 4)
    plt.title('Poisson Noise')
    plt.imshow(Image.fromarray(noisy_poisson.astype(np.uint8)))
    plt.axis('off')

    plt.subplot(2, 3, 5)
    plt.title('Speckle Noise')
    plt.imshow(Image.fromarray(noisy_speckle.astype(np.uint8)))
    plt.axis('off')

    plt.tight_layout()
    plt.show()
