import torch

def generate_mask(attention_map, threshold=0.00025):
    # Threshold the cumulative attention maps
    print("max of attention_map: ", torch.max(attention_map))
    print("min of attention_map: ", torch.min(attention_map))
    print("mean of attention_map: ", torch.mean(attention_map))
    # TODO: devise a strategy to threshold the attention_map
    threshold =  torch.mean(attention_map)

    mask = (attention_map > threshold).float()
    return mask

def intersect_masks(masks):
    mask = masks[0]
    
    # Iterate over the rest of the tensors and multiply them with the mask
    for m in masks[1:]:
        mask *= m
        
    return mask

def latent_space_manipulation(latents, noised_latent_t, attention_maps, is_gradients=False):

    if is_gradients:
        # Generate masks for each attention map
        masks = [generate_mask(attention_map) for attention_map in attention_maps]
        mask = intersect_masks(masks)

        # Find indices where mask is 0
        zero_indices = (mask == 0).nonzero()

        # Replace values in latents with corresponding values from noised_latent_t
        for idx in zero_indices:
            latents[0, :, idx[2], idx[3]] = noised_latent_t[0, :, idx[2], idx[3]]
    else:
        mask = generate_mask(attention_maps)

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

