import torch
import torch.nn as nn

def train_conditional_diffusion(net, train_dataloader, noise_scheduler, optimizer, loss_fn, epochs, device):
  """
  Trains a conditional denoising diffusion model.

  Args:
      net: The conditional denoising diffusion model to train.
      train_dataloader: DataLoader for the training dataset.
      noise_scheduler: Noise scheduler for adding and removing noise.
      optimizer: Optimizer for updating model parameters.
      loss_fn: Loss function for calculating the error.
      epochs: Number of training epochs.
      device: Device to run the training on (e.g., 'cuda' or 'cpu').
  """
  losses = []
  for epoch in range(epochs):
      for x, y in train_dataloader:
          # Get some data and prepare the corrupted version
          x = x.to(device) * 2 - 1  # Data on the GPU (mapped to (-1, 1))
          y = y.to(device)
          noise = torch.randn_like(x)
          timesteps = torch.randint(0, noise_scheduler.num_train_timesteps - 1, (x.shape[0],)).long().to(device)
          noisy_x = noise_scheduler.add_noise(x, noise, timesteps)

          # Get the model prediction
          pred = net(noisy_x, timesteps, y)  # Pass in the labels y

          # Calculate the loss
          loss = loss_fn(pred, noise)  # How close is the output to the noise

          # Backprop and update the params:
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()

          # Store the loss for later
          losses.append(loss.item())

      # Print out the average of the loss values for this epoch:
      avg_loss = sum(losses[-len(train_dataloader):]) / len(train_dataloader)
      print(f"Finished epoch {epoch}. Average loss: {avg_loss:05f}")


def train_denoise_diffusion(model, train_dataloader, noise_scheduler, optimizer, criterion, epochs, device):
  """
  Trains a denoising diffusion model.

  Args:
      model: The denoising diffusion model to train.
      train_dataloader: DataLoader for the training dataset.
      noise_scheduler: Noise scheduler for adding and removing noise.
      optimizer: Optimizer for updating model parameters.
      criterion: Loss function for calculating the error.
      epochs: Number of training epochs.
      device: Device to run the training on (e.g., 'cuda' or 'cpu').
  """
  losses = []
  for epoch in range(epochs):
      for x, _ in train_dataloader:
          # Get some data and prepare the corrupted version
          x = x.to(device)
          noise = torch.randn_like(x)
          timesteps = torch.randint(
              0, noise_scheduler.num_train_timesteps, (x.shape[0],), device=device
          )
          noisy_x = noise_scheduler.add_noise(x, noise, timesteps)

          # Get the model prediction
          pred = model(noisy_x)

          # Calculate the loss
          loss = criterion(pred, noise)  # Predict the noise instead of x

          # Backprop and update the params:
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()

          # Store the loss for later
          losses.append(loss.item())

      # Print our the average of the loss values for this epoch:
      avg_loss = sum(losses[-len(train_dataloader) :]) / len(train_dataloader)
      print(f"Epoch [{epoch+1}/{epochs}], Average loss: {avg_loss:05f}")