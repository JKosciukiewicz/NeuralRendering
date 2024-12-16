import torch
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

def evaluate_autoencoder(model, data_loader, add_noise_fn, visualize=True):
    """
    Evaluate the performance of an autoencoder using PSNR and SSIM metrics.

    Parameters:
        model (torch.nn.Module): The trained autoencoder model.
        data_loader (torch.utils.data.DataLoader): DataLoader for the test dataset.
        add_noise_fn (function): Function to add noise to input images.
        visualize (bool): Whether to visualize noisy, original, and reconstructed images.

    Returns:
        tuple: Average PSNR and SSIM across the dataset.
    """
    model.eval()
    total_psnr = 0
    total_ssim = 0
    num_batches = 0

    with torch.no_grad():
        for images, _ in data_loader:
            # Add noise to test images
            noisy_images = add_noise_fn(images)
            # Get reconstructed images
            outputs = model(noisy_images)

            # Calculate PSNR and SSIM for the batch
            batch_psnr, batch_ssim = calculate_metrics(images, outputs)
            total_psnr += batch_psnr
            total_ssim += batch_ssim
            num_batches += 1

            # Visualize only the first batch
            if visualize and num_batches == 1:
                visualize_reconstructions(noisy_images, images, outputs)

    # Calculate average metrics
    avg_psnr = total_psnr / num_batches
    avg_ssim = total_ssim / num_batches

    print(f"Average PSNR: {avg_psnr:.2f}")
    print(f"Average SSIM: {avg_ssim:.4f}")

    return avg_psnr, avg_ssim

def calculate_metrics(original, reconstructed):
    """
    Calculate PSNR and SSIM for a batch of images.

    Parameters:
        original (torch.Tensor): Original images (ground truth).
        reconstructed (torch.Tensor): Reconstructed images from the model.

    Returns:
        tuple: Mean PSNR and SSIM for the batch.
    """
    psnr_values = []
    ssim_values = []
    for i in range(original.size(0)):
        # Convert tensors to numpy arrays
        orig_img = original[i].squeeze().cpu().numpy()
        recon_img = reconstructed[i].squeeze().cpu().numpy()

        # Compute PSNR and SSIM
        psnr_values.append(psnr(orig_img, recon_img, data_range=2.0))  # Data range is [-1, 1] -> 2.0
        ssim_values.append(ssim(orig_img, recon_img, data_range=2.0))

    return np.mean(psnr_values), np.mean(ssim_values)

def visualize_reconstructions(noisy_images, original_images, reconstructed_images, n=5):
    """
    Visualize noisy, original, and reconstructed images.

    Parameters:
        noisy_images (torch.Tensor): Noisy input images.
        original_images (torch.Tensor): Original ground truth images.
        reconstructed_images (torch.Tensor): Reconstructed images from the model.
        n (int): Number of images to visualize.
    """
    plt.figure(figsize=(15, 6))
    for i in range(n):
        # Noisy
        plt.subplot(3, n, i + 1)
        plt.imshow(noisy_images[i].squeeze().numpy(), cmap='gray')
        plt.axis('off')
        plt.title("Noisy")

        # Original
        plt.subplot(3, n, i + 1 + n)
        plt.imshow(original_images[i].squeeze().numpy(), cmap='gray')
        plt.axis('off')
        plt.title("Original")

        # Reconstructed
        plt.subplot(3, n, i + 1 + 2 * n)
        plt.imshow(reconstructed_images[i].squeeze().numpy(), cmap='gray')
        plt.axis('off')
        plt.title("Reconstructed")

    plt.show()

