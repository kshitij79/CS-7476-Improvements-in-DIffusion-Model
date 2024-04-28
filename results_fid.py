import cv2
import os
import numpy as np
from torchmetrics.image.fid import FrechetInceptionDistance
import torch
import json

# Define a function to calculate FID score
def get_fid(images1, images2):
    images1 = torch.tensor(images1, dtype=torch.uint8)
    images2 = torch.tensor(images2, dtype=torch.uint8)

    # Add batch dimension
    images1 = images1.unsqueeze(0)
    images2 = images2.unsqueeze(0)

    # Batch_size * C * H * W -> Batch_size * H * W * C
    images1 = images1.permute(0, 3, 1, 2)
    images2 = images2.permute(0, 3, 1, 2)

    # Repeat images for FID calculation
    images1 = images1.repeat(2, 1, 1, 1)
    images2 = images2.repeat(2, 1, 1, 1)

    # Initialize FID metric
    fid = FrechetInceptionDistance(feature=64)
    fid.update(images1, real=True)
    fid.update(images2, real=False)
    score = fid.compute()
    return score.item()

# Function to evaluate generated images
def evaluate_generated_images(experiments, original_folder, generated_folder):
    fid_scores = {}

    # Loop through each experiment category
    for experiment in experiments:
        category = experiment["category"]
        category_fid_scores = {}
        
        # Loop through each subcategory in the category
        for subcategory_data in experiment["subcategories"]:
            subcategory = subcategory_data["subcategory"]
            prompts = subcategory_data["prompts"]
            subcategory_fid_scores = {}

            # Loop through each prompt
            for prompt in prompts:
                prompt_fid_scores = {}

                # Loop through each reference image in the original folder
                for filename in os.listdir(os.path.join(original_folder, category, subcategory)):
                    if filename.endswith(".jpg"):
                        reference_image_path = os.path.join(original_folder, category, subcategory, filename)
                        reference_image = cv2.imread(reference_image_path)

                        # Construct the filename of the generated image based on the prompt
                        print(filename)
                        generated_image_filename = filename
                        generated_image_path = os.path.join(generated_folder, category, subcategory, prompt, generated_image_filename)

                        # Load the generated image
                        generated_image = cv2.imread(generated_image_path)

                        # Calculate FID score and store it in the dictionary
                        fid_score = get_fid(reference_image, generated_image)
                        prompt_fid_scores[filename] = fid_score

                subcategory_fid_scores[prompt] = prompt_fid_scores
            
            category_fid_scores[subcategory] = subcategory_fid_scores
        
        fid_scores[category] = category_fid_scores

    return fid_scores

# Load experiments from JSON
with open("experiments.json", "r") as f:
    experiments = json.load(f)

# Paths to original and generated image folders
original_folder = "./DATA-1"
generated_folder = "./generated_images"

# Evaluate generated images
fid_scores = evaluate_generated_images(experiments["experiments"], original_folder, generated_folder)

# Transform the fid_scores dictionary to include prompts as keys
# and fid scores as values
results = {}
for category, subcategories in fid_scores.items():
    results[category] = {}
    for subcategory, prompts in subcategories.items():
        for prompt, scores in prompts.items():
            if prompt not in results[category]:
                results[category][prompt] = []
            results[category][prompt].append(scores)

# Save FID scores to results_fid.json
with open("results_fid.json", "w") as f:
    json.dump(results, f, indent=4)

print("FID scores saved to results_fid.json")
