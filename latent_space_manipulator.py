import torch

def generate_mask(cumulative_attention_maps, threshold=0.00025):
    # Threshold the cumulative attention maps
    print("max of cumulative_attention_maps: ", torch.max(cumulative_attention_maps))
    print("min of cumulative_attention_maps: ", torch.min(cumulative_attention_maps))
    print("mean of cumulative_attention_maps: ", torch.mean(cumulative_attention_maps))
    # TODO: devise a strategy to threshold the cumulative_attention_maps
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
    # TODO: devise a strategy to select timesteps for manipulating latent space
    timesteps = [sampler.timesteps[5], sampler.timesteps[10], sampler.timesteps[15]]
    return timesteps