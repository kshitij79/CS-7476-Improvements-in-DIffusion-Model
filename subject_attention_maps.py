from torch import tensor, reshape, sum, zeros
import torch.nn.functional as F
import numpy as np

def extract_subject_attention_maps(attention_maps, subject_info):
    subject_attention_maps = {}

    for subject, idx in subject_info.items():
        subject_attention_maps[subject] = {}
        for attention_key in attention_maps.keys():
            attention_map = attention_maps[attention_key]
            subject_attention_map = attention_map[0:1, :, :, idx]
            # only keep 0th batch as it captues conditional info 0:, 0, : only keep the first batch
            subject_attention_map = subject_attention_map[0:, :, :]
            # we reshape the attention map to 2D
            # Reshape the attention map to a 2D square form
            h, w = int(np.sqrt(subject_attention_map.shape[2])), int(np.sqrt(subject_attention_map.shape[2]))
            batch_size, n_heads = subject_attention_map.shape[0], subject_attention_map.shape[1]
            square_attention_map = zeros((batch_size, n_heads, h, w)) 
            for i in range(batch_size):
                for j in range(n_heads):
                    # we keep batch_size and n_heads as is
                    square_attention_map[i, j] = reshape(subject_attention_map[i, j], (h, w))

            subject_attention_maps[subject][attention_key] = square_attention_map

    return subject_attention_maps


# def upscale_attention_map(attention_map, target_size):
#     # Upscale the attention map to the target size
#     scaled_attention_map = tensor.repeat(tensor.repeat(attention_map, target_size, axis=2), target_size, axis=3)
#     return scaled_attention_map

def upscale_attention_map(attention_map, target_size):
    scaled_attention_map = F.interpolate(attention_map, size=(target_size, target_size), mode='nearest')
    return scaled_attention_map

def normalize_attention_map(attention_map):
    # Normalize the attention map across the square dimension with all points summing to 1
    normalized_attention_map = attention_map / attention_map.sum(axis=(2, 3), keepdims=True)
    return normalized_attention_map

def cumulative_attention_map(subject_attention_maps, target_size=64):
    cumulative_attention_maps = {}
    
    for subject, attention_maps in subject_attention_maps.items():
        # 0 key have the maximum height and width
        batch_size, n_heads, target_size, _ = attention_maps[list(attention_maps.keys())[0]].shape

        cumulative_map = zeros((batch_size, 1, target_size, target_size))
        for attention_key, attention_map in attention_maps.items():
            # h = attention_map.shape[2]
            # target_scale = target_size // h
            scaled_attention_map = upscale_attention_map(attention_map, target_size)
            # we sum along axis 1 reducing it from n_heads to 1
            scaled_attention_map = scaled_attention_map.sum(dim=1, keepdim=True)
            cumulative_map += scaled_attention_map

        cumulative_map = normalize_attention_map(cumulative_map)
        cumulative_attention_maps[subject] = tensor(cumulative_map)

    return cumulative_attention_maps    

def cumulative_subject_attention_map(cumulative_attention_maps, target_size=64):
    # tensor of shape 1, 1, 64, 64
    cumulative_subject_attention_maps = zeros((1, 1, target_size, target_size))
    # we will add attention maps of all subjects
    for subject, idx in cumulative_attention_maps.items():
        cumulative_subject_attention_maps += cumulative_attention_maps[subject]

    # we normalize the cumulative attention map
    cumulative_subject_attention_maps = normalize_attention_map(cumulative_subject_attention_maps)
    return tensor(cumulative_subject_attention_maps)




       


    # cumulative_attention_maps = {}
    # batch_size, n_heads, seq_len_q, seq_len_kv = attention_maps[list(attention_maps.keys())[0]].shape

    # # Iterate through each subject
    # for subject, idx in subject_info.items():
    #     cumulative_map = np.zeros((batch_size, n_heads, target_size, target_size))

    #     # Iterate through each attention key
    #     for attention_key, attention_map in attention_maps.items():
    #         subject_attention_map = attention_map[0, :, :, idx]  # Extract subject-specific attention map

    #         # Upscale attention map to target size
    #         scaled_attention_map = upscale_attention_map(subject_attention_map, target_size)

    #         # Add scaled attention map to cumulative map
    #         cumulative_map += scaled_attention_map

    #     # Normalize the cumulative attention map
    #     cumulative_map = normalize_attention_map(cumulative_map)

    #     cumulative_attention_maps[subject] = cumulative_map

    # return cumulative_attention_maps