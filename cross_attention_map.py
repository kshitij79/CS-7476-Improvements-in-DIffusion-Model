import torch
from torchvision import utils
import os
import matplotlib.pyplot as plt
import numpy as np

def visualize_attention_maps(attentions, subject_info, time, folder='images'):
    os.makedirs(folder, exist_ok=True)
    batch_size, n_heads, seq_len_q, seq_len_kv = attentions.size()
    h, w = int(np.sqrt(seq_len_q)), int(np.sqrt(seq_len_q))
    
    for subject, idx in subject_info.items():
        for head in range(n_heads):
            attention_map = attentions[:, head, :, :].detach().cpu().numpy()  # Extract attention map for the subject and head
            attention_map = attention_map[:, :, idx] # Extract attention for the subject token
            
            # Normalize attention weights for better visualization
            epsilon = np.finfo(float).eps  # Small constant to avoid division by zero
            attention_map -= attention_map.min()  # Compute the minimum value
            attention_map /= attention_map.max() + epsilon  # Normalize
            
            # Reshape the attention map to a 2D square form

            square_attention_map = np.zeros((batch_size, h, w))
            for i in range(batch_size):
                square_attention_map[i] = np.reshape(attention_map[i], (h, w))
            
            # Create the plot
            plt.imshow(square_attention_map[0], cmap='hot', interpolation='nearest')
            plt.colorbar()
            plt.title(f'Attention Map - Subject: {subject}, Head: {head}')
            plt.xlabel('Key/Value Sequence Position')
            plt.ylabel('Query Sequence Position')
            plt.tight_layout()
            
            # Save the plot as an image
            filename = f'{folder}/attention_map_{subject}_head{head}_time{time}.png'
            plt.savefig(filename)
            plt.close()

def visualize_cumulative_map(cumulative_attention_map, time):
    """
    Visualizes the cumulative attention map at a specific time index.
    
    Parameters:
        cumulative_attention_map (dict): Cumulative attention maps for different subjects.
        time (int): Index of the time step to visualize.
    """
    # Assuming cumulative_attention_map is a dictionary with subject names as keys
    for subject, attention_map in cumulative_attention_map.items():
        # Assuming attention_map is a 4D tensor with shape [batch_size, 1, target_size, target_size]
        # Extract attention map for the specified time index
        attention_map_time = attention_map[0][0]
        # Plot the attention map
        plt.imshow(attention_map_time, cmap='hot', interpolation='nearest')
        plt.title(f'Cumulative Attention Map for {subject} at Time Step {time}')
        plt.colorbar()
        plt.show()   

# Example usage:
# Assuming attentions is the attention tensor of shape (Batch_Size, H, Seq_Len_Q, Seq_Len_KV)
# and subject_info is a dictionary containing subject tokens and their indices
# visualize_attention_maps(attentions, subject_info)
