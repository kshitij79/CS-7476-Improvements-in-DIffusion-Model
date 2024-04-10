# from torchviz import make_dot
# import torch
# import model_loader

# import sys
# sys.setrecursionlimit(5000)
# WIDTH = 512
# HEIGHT = 512
# LATENTS_WIDTH = WIDTH // 8
# LATENTS_HEIGHT = HEIGHT // 8
# seed = 42

# device = "cuda" if torch.cuda.is_available() else "cpu"
# model_file = "./data/models--runwayml--stable-diffusion-v1-5/snapshots/1d0c4ebf6ff58a5caecab40fa1406526bca4b5b9/v1-5-pruned-emaonly.ckpt"

# # Load models and tokenizer
# models = model_loader.preload_models_from_standard_weights(model_file, device)

# # Initialize input tensors
# latents_shape = (1, 4, LATENTS_HEIGHT, LATENTS_WIDTH)
# latents = torch.randn(latents_shape, device=device)
# uncond_context = models["clip"](torch.zeros(1, 77, device=device))

# # Define function to get time embedding
# def get_time_embedding(timestep):
#     freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160)
#     x = torch.tensor([timestep], dtype=torch.float32, device=device)[:, None] * freqs[None]
#     return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)

# # Instantiate Diffusion model
# model = models["diffusion"]
# model.to(device)
# model.eval()  # Ensure evaluation mode

# # Perform forward pass
# time_embedding = get_time_embedding(0)
# outputs = model(latents, uncond_context, time_embedding)

# # Visualize the computation graph
# make_dot(outputs, params=dict(model.named_parameters())).render("computation_graph", format="png")

from torchviz import make_dot
import torch
import model_loader
import numpy as np
import sys
sys.setrecursionlimit(2000)

WIDTH = 512
HEIGHT = 512
LATENTS_WIDTH = WIDTH // 8
LATENTS_HEIGHT = HEIGHT // 8
seed = 42

device = "cuda" if torch.cuda.is_available() else "cpu"
model_file = "./data/models--runwayml--stable-diffusion-v1-5/snapshots/1d0c4ebf6ff58a5caecab40fa1406526bca4b5b9/v1-5-pruned-emaonly.ckpt"

# Load models and tokenizer
models = model_loader.preload_models_from_standard_weights(model_file, device)

# Instantiate Diffusion model
diffusion_model = models["diffusion"]
diffusion_model.to(device)
diffusion_model.eval()  # Ensure evaluation mode

# Define function to get time embedding
def get_time_embedding(timestep):
    # Shape: (160,)
    freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160) 
    # Shape: (1, 160)
    x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]
    # Shape: (1, 160 * 2)
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)

# Sample input tensors
latents_shape = (1, 4, LATENTS_HEIGHT, LATENTS_WIDTH)
latents = torch.randn(latents_shape, device=device)
uncond_context = models["clip"](torch.zeros(1, 77, device=device))

# Perform forward pass through Unet part of the diffusion model
timesteps = torch.from_numpy(np.arange(0, 50)[::-1].copy())
timestep = timesteps[0]
time_embedding = get_time_embedding(timestep)
unet_output = diffusion_model(latents, uncond_context, time_embedding)

# Visualize the computation graph for the Unet part only
make_dot(unet_output, params=dict(diffusion_model.named_parameters())).render("unet_computation_graph", format="png")

