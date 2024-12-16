import torch
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
from diffusers import DDPMScheduler  # Assuming you're using a DDPM scheduler

def corrupt(x, amount):
    """Corrupt the input `x` by mixing it with noise according to `amount`"""
    noise = torch.rand_like(x)
    amount = amount.view(-1, 1, 1, 1)  # Sort shape so broadcasting works
    return x * (1 - amount) + noise * amount

def evaluate_diffusion_model(model, data_loader, device, visualize=True):
    """
    Evaluate the performance of a diffusion model using PSNR and SSIM metrics.

    Parameters:
        model (torch.nn.Module): The trained diffusion model.
        data_loader (torch.utils.data.DataLoader): DataLoader for the test dataset.
        noise_scheduler (diffusers.DDPMScheduler): Noise scheduler used in the diffusion process.
        device (torch.device): Device to run the evaluation on (e.g., 'cuda' or 'mps').
        visualize (bool): Whether to visualize original and generated images.

    Returns:
        tuple: Average PSNR and SSIM across the dataset.
    """
    model.eval()
    total_psnr = 0
    total_ssim = 0
    num_batches = 0

    with torch.no_grad():
        for images, _ in data_loader:
            images = images.to(device)

            # Sample noise and add it to the images
            noise_amount = torch.rand(images.shape[0]).to(device)
            noisy_images= corrupt(images, noise_amount)  # Add noise to the images

            # Generate images from the diffusion model
            generated_images = model(noisy_images)

            # Calculate PSNR and SSIM for the batch
            batch_psnr, batch_ssim = calculate_metrics(noisy_images, generated_images)
            total_psnr += batch_psnr
            total_ssim += batch_ssim
            num_batches += 1

            # Visualize only the first batch
            if visualize and num_batches == 1:
                visualize_diffusion_samples(noisy_images, generated_images)

    # Calculate average metrics
    avg_psnr = total_psnr / num_batches
    avg_ssim = total_ssim / num_batches

    print(f"Average PSNR: {avg_psnr:.2f}")
    print(f"Average SSIM: {avg_ssim:.4f}")



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


def visualize_diffusion_samples(original_images, generated_images, n=5):
    """
    Visualize original and generated images from the diffusion model.

    Parameters:
        original_images (torch.Tensor): Original ground truth images.
        generated_images (torch.Tensor): Generated images from the diffusion model.
        n (int): Number of images to visualize.
    """
    plt.figure(figsize=(15, 6))
    for i in range(n):
        # Original
        plt.subplot(2, n, i + 1)
        plt.imshow(original_images[i].squeeze().cpu().numpy(), cmap='gray')
        plt.axis('off')
        plt.title("Original")

        # Generated
        plt.subplot(2, n, i + 1 + n)
        plt.imshow(generated_images[i].squeeze().cpu().numpy(), cmap='gray')
        plt.axis('off')
        plt.title("Generated")

    plt.show()