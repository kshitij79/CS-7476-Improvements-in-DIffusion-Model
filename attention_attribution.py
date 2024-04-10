import torch
import torch.nn.functional as F

def calculate_contributions(attention_gradients, subject_info):
    contributions = {}
    
    # {subject_key: {dim_key: {index_key: contribution score}}}
    for subject, index in subject_info.items():
        subject_contributions = {}
        
        for key, attention_map in attention_gradients.items():
            contribution_list = {}
            for map_key, gradients in attention_map.items():
                absolute_gradients = torch.abs(gradients[0:1, :, :, int(index)])
                contribution = torch.sum(absolute_gradients).item()
                contribution_list[map_key] = contribution
            subject_contributions[key] = contribution_list

        contributions[subject] = subject_contributions
    
    return contributions

def weighted_sum_attention_maps(attention_maps, contributions):
    """
    Computes the weighted sum of attention maps.
    
    Args:
        attention_maps (dict): A dictionary containing attention maps for different subjects.
        weights (dict): A dictionary containing weights for each subject.
    
    Returns:
        dict: A dictionary containing the weighted sum of attention maps for each subject.
    """
    weighted_attention_maps = {}
    
    # {subject_key: {dim_key: {index_key: contribution score}}}
    for subject, weights in contributions.items():
        # {dim_key, {index_key: contribution score} }
        weighted_attention_maps[subject] = {}
        for key, contribution in weights.items():
            # {index_key: contribution score}
            attention_map = attention_maps[key]['0']
            # we remove the batch dimension
            attention_map = attention_map[0:1, :, :, :]
            weighted_sum = torch.zeros_like(attention_map)
            # normalize the contribution scores
            total_contribution = sum(contribution.values())
            for map_key, value in contribution.items():
                value = value / total_contribution
                # we remove the batch dimension
                attention_map = attention_maps[key][map_key][0:1, :, :, :]

                weighted_sum = attention_maps[key][map_key] * value + weighted_sum
            # {subject_key: {dim_key: weighted sum of attention maps}}
            weighted_attention_maps[subject][key] = weighted_sum

    return weighted_attention_maps

def normalize_attention_map(attention_map):
    # Normalize the attention map across the square dimension with all points summing to 1
    normalized_attention_map = attention_map / attention_map.sum(axis=(2, 3), keepdims=True)
    return normalized_attention_map

def upscale_attention_map(attention_map, target_size):
    scaled_attention_map = F.interpolate(attention_map, size=(target_size, target_size), mode='nearest')
    return scaled_attention_map

def scale_weighted_attention_maps(weighted_attention_maps, target_size=64):
    scaled_attention_maps = {}
    
    for subject, attention_maps in weighted_attention_maps.items():
        scaled_attention_maps[subject] = {}
        for key, attention_map in attention_maps.items():
            # upscale the attention map
            scaled_attention_map = upscale_attention_map(attention_map, target_size)

            # normalize the attention map
            attention_map = normalize_attention_map(attention_map)
            scaled_attention_maps[subject][key] = scaled_attention_map
    
    return scaled_attention_maps


def attribution_pipeline(attention_gradients, attention_map, subject_info, target_size=64):

    contributions = calculate_contributions(attention_gradients, subject_info)
    weighted_attention_maps = weighted_sum_attention_maps(attention_map, contributions)
    scaled_attention_maps = scale_weighted_attention_maps(weighted_attention_maps, target_size)
    
    return scaled_attention_maps

def attention_map_dict_to_list(attention_map_dict):
    attention_map_list = []
    for subject, attention_map in attention_map_dict.items():
        for key, value in attention_map.items():
            attention_map_list.append(value)
    return attention_map_list


            