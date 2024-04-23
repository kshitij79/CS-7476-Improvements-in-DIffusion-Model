import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import peak_local_max
from sklearn.cluster import DBSCAN

def gaussian_kernel(size, sigma, channels):
    """Generate a 2D Gaussian kernel expanded across specified channels for grouped convolution."""
    coords = torch.arange(size)
    coords -= size // 2

    g = coords**2
    g = g.view(1, -1) + g.view(-1, 1)
    g = torch.exp(-g / (2 * sigma ** 2))

    g /= g.sum()
    # Expand to the number of channels: each channel uses the same kernel
    return g.view(1, 1, size, size).repeat(channels, 1, 1, 1)

def create_dilation_kernel(kernel_size, channels):
    """Generate a square kernel for morphological dilation."""
    kernel = torch.ones(kernel_size, kernel_size)
    kernel = kernel.view(1, 1, kernel_size, kernel_size).repeat(channels, 1, 1, 1)
    return kernel

# function to compress attention maps, first it filter the subjects of interest, then it sums the gradients for each subject
def compress_attention_gradients(attention_gradients, subject_info):
    """
    Compress the attention gradients by filtering the subjects of interest and summing the gradients for each subject.
    input: attention_gradient, subject_info
    return: summed_attention_maps
    """
    summed_attention_gradients = []
    # print("attention_gradients", len(attention_gradients), attention_gradients[0].shape)
    for attention_gradient in attention_gradients[0]:
        summed_attention_gradient = {}
        for subject, index in subject_info.items():
            gradient = attention_gradient[0, :, :, int(index)]
            thresholded_gradient = torch.where(gradient > 0, gradient, torch.tensor(0.0))
            summed_attention_gradient[subject] = torch.sum(thresholded_gradient)
        summed_attention_gradients.append(summed_attention_gradient)       
    # print(len(summed_attention_gradients))
    return summed_attention_gradients

def importance_score(attention_map, attention_gradients, subject_info):
    """
    Compress the attention gradients by filtering the subjects of interest and summing the gradients for each subject.
    input: attention_gradient, subject_info
    return: summed_attention_maps
    """
    summed_attention_gradients = []
    # print("attention_gradients", len(attention_gradients), attention_gradients[0].shape)
    for attention_gradient in attention_gradients[0]:
        summed_attention_gradient = {}
        for subject, index in subject_info.items():
            gradient = attention_gradient[0, :, :, int(index)]
            attention = attention_map[0, :, :, int(index)]
            thresholded_gradient = torch.where(gradient > 0, gradient, torch.tensor(0.0))
            # in summed_attention_gradient, we store the sum of hadamard product of attention_map and thresholded_gradient
            # https://arxiv.org/pdf/2204.11073.pdf motivated from Grad SAM
            summed_attention_gradient[subject] = torch.sum(attention*thresholded_gradient)
        summed_attention_gradients.append(summed_attention_gradient)       
    # print(len(summed_attention_gradients))
    return summed_attention_gradients

def parse_key(key):
    components = key.split('_')
    parsed = {}
    for component in components:
        if component.startswith('k'):
            parsed['k'] = int(component[1:])
        elif component.startswith('b'):
            parsed['b'] = int(component[1:])
        elif component.startswith('c'):
            parsed['c'] = int(component[1:])
        elif component.startswith('i'):
            parsed['i'] = int(component[1:])
        elif component.startswith('j'):
            parsed['j'] = int(component[1:])
        elif component.startswith('scores'):
            parsed['scores'] = component[6:]
    return parsed

def score_map(attention_gradients, output):
    """
    Score the attention map by filtering the subjects of interest and summing the gradients for each subject.
    input: attention_gradients, output
    return: score_map
    """
    score_map = {}

    for layer, gradients_dict in attention_gradients.items():
        if score_map.get(layer) is None:
            score_map[layer] = torch.zeros_like(output)
        for key, gradients in gradients_dict.items():
            parsed_key = parse_key(key)
            k = parsed_key['k']
            b = parsed_key['b']
            c = parsed_key['c']
            i = parsed_key['i']
            j = parsed_key['j'] 

            for subject, score in gradients.items():
                score_map[layer][b, c, i, j] += score
        score_map = score_map_dict_to_list(score_map)
    return score_map

def score_map_with_subjects(attention_gradients, output):
    """
    Score the attention map by filtering the subjects of interest and summing the gradients for each subject.
    input: attention_gradients, output
    return: score_map
    """
    score_map = {}

    for layer, gradients_dict in attention_gradients.items():
        k = 0  # Initialize k value
        for key, gradients in gradients_dict.items():
            parsed_key = parse_key(key)
            k = parsed_key['k']
            b = parsed_key['b']
            c = parsed_key['c']
            i = parsed_key['i']
            j = parsed_key['j'] 

            for subject, score in gradients.items():
                if score_map.get(str(layer) + "_" + str(k)) is None:
                    score_map[str(layer) + "_" + str(k)] = {}
                if score_map[str(layer) + "_" + str(k)].get(subject) is None:
                    score_map[str(layer) + "_" + str(k)][subject] = torch.zeros_like(output)
                score_map[str(layer) + "_" + str(k)][subject][b, c, i, j] = score

    score_map = score_map_dict_to_list(score_map)
    return score_map

def score_map_dict_to_list(score_map_dict):
    """
    Convert the score map dictionary to a list.
    input: score_map_dict
    return: score_map_list
    """
    score_map_list = []
    for layer, score_map in score_map_dict.items():
        score_map_list.append(score_map)

    return score_map_list

def create_mask(score_map):
    """
    Create a mask by thresholding the score map.
    input: score_map, threshold
    return: mask
    """
    print("max of attention_map: ", torch.max(score_map))
    print("min of attention_map: ", torch.min(score_map))
    print("mean of attention_map: ", torch.mean(score_map))
    print("Shape of score_map:", score_map.size())
    batch_size, channel, height, width = score_map.size()

    # if isinstance(threshold, float):
    #     threshold = torch.tensor([threshold] * channel, device=score_map.device)

    masks = []
    for b in range(batch_size):
        mask_per_batch = []
        for c in range(channel):
            window_score_map = score_map[b][c]
            window_threshold = torch.mean(score_map[b][c])
            print("window_threshold:", c, "value", window_threshold)
            window_mask = (window_score_map > window_threshold).float().unsqueeze(0)
            mask_per_batch.append(window_mask)
        masks.append(torch.cat(mask_per_batch, dim=0))
    mask = torch.cat(masks, dim=0)
    mask = mask.unsqueeze(0)
    return mask

def create_mask_from_subjects(score_map_list, timestamp):
    """
    Create a mask by thresholding the score map after applying Gaussian smoothing.
    
    Input:
        score_map_list: Dictionary of score maps
        timestamp: int tensor
    Return:
        mask: Tensor representing the union of thresholded score maps
    """
    device = list(score_map_list.values())[0].device  # Get the device of the first score map
    batch_size, channel, height, width = list(score_map_list.values())[0].size()
    mask = torch.zeros(batch_size, channel, height, width, device=device)

    # Define Gaussian kernel (e.g., size=5, sigma=1.0 for this example)
    kernel_size = 3
    sigma = 6.0
    kernel = gaussian_kernel(kernel_size, sigma, channel).to(device)

    #
    dilation_kernel = create_dilation_kernel(kernel_size, channel).to(device)

    num_iterations = 3

    for subject, score_map in score_map_list.items():
        # Apply Gaussian smoothing
        score_map_smooth = F.conv2d(score_map, kernel, padding=kernel_size//2, groups=channel)
        
        # Apply morphological dilation
        score_map_dilated = score_map
        for _ in range(num_iterations):
            score_map_smooth = F.conv2d(score_map_dilated, dilation_kernel, padding=kernel_size//2, groups=channel)

        # Calculate thresholds
        std_threshold = torch.std(score_map_smooth)
        threshold = torch.mean(score_map_smooth) + 0.5 * std_threshold # * (max_timestamp - timestamp) * 0.012
        print(subject, " ", threshold, torch.max(score_map_smooth), torch.min(score_map_smooth), std_threshold)
        
        # Create window mask
        window_mask = (score_map_smooth > threshold).float().to(device)  # Move window_mask to the same device
        
        # Taking union of masks for all subjects
        mask = torch.max(mask, window_mask)

    return mask

def intersect_masks(masks):
    """
    Intersect the masks.
    input: masks
    return: mask
    """
    mask = masks[0]
    
    for m in masks[1:]:
        mask *= m
        
    return mask

def latent_space_manipulation(latents, noised_latent_t, score_maps):
    """
    Manipulate the latent space by replacing the values in the latents tensor with the corresponding values from the noised_latent_t tensor.
    input: latents, noised_latent_t, score_maps
    return: latents
    """
    masks = [create_mask(score_map) for score_map in score_maps]
    # print(len(masks))
    mask = intersect_masks(masks)
    zero_indices = (mask == 0).nonzero()
    # print(zero_indices)
    for idx in zero_indices:
        latents[0, idx[1], idx[2], idx[3]] = noised_latent_t[0, idx[1], idx[2], idx[3]]

    return latents

def latent_space_manipulation_subjectwise(latents, noised_latent_t, score_maps, timestamp):
    """
    Manipulate the latent space by replacing the values in the latents tensor with the corresponding values from the noised_latent_t tensor.
    input: latents, noised_latent_t, score_maps
    return: latents
    """
    masks = [create_mask_from_subjects(score_map, timestamp) for score_map in score_maps]
    
    mask = intersect_masks(masks)
    zero_indices = (mask == 0).nonzero()
    zero_len = len(zero_indices)
    save_path = 'images/mask_heatmap_{zero_len}.png'
    visualize_heatmap(mask, save_path)
    
    print(len(zero_indices))
    for idx in zero_indices:
        latents[0, idx[1], idx[2], idx[3]] = noised_latent_t[0, idx[1], idx[2], idx[3]]

    return latents    

def visualize_heatmap(mask, save_path):
    """
    Visualize the mask as a heatmap and save it as an image.
    Args:
    - mask: The mask to visualize.
    - save_path: The path to save the visualization image.
    """
    for i in range(mask.shape[1]):
        heatmap = mask[0][i].cpu().numpy()  # Assuming mask is a torch tensor
        plt.imshow(heatmap, cmap='hot', interpolation='nearest')
        plt.colorbar()
        plt.axis('off')
        plt.savefig(f"{save_path}_{i}.png")
        plt.close()
