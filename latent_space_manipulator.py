import torch

def generate_mask(cumulative_attention_maps, threshold=0.5):
    # Threshold the cumulative attention maps
    mask = (cumulative_attention_maps > threshold).float()
    return mask

def latent_space_manipulation(latents, noised_latent_t, cumulative_attention_maps):

    # Generate mask
    mask = generate_mask(cumulative_attention_maps)

    # Find indices where mask is 0
    zero_indices = (mask == 0).nonzero()

    # Replace values in latents with corresponding values from noised_latent_t
    for idx in zero_indices:
        latents[0, :, idx[2], idx[3]] = noised_latent_t[0, :, idx[2], idx[3]]

    return latents

def timestamps_to_manipulate(sampler):
    # control which other noised latents are needed for particular timesteps
    timesteps = [sampler.timesteps[1], sampler.timesteps[2], sampler.timesteps[5]]
    return timesteps