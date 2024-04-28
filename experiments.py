import model_loader
import pipeline
from PIL import Image
from pathlib import Path
from transformers import CLIPTokenizer
import torch
import importlib
import json
import os

importlib.reload(model_loader)
importlib.reload(pipeline)

DEVICE = "cpu"

ALLOW_CUDA = True
ALLOW_MPS = False # True

if torch.cuda.is_available() and ALLOW_CUDA:
    DEVICE = "cuda"
elif (torch.has_mps or torch.backends.mps.is_available()) and ALLOW_MPS:
    DEVICE = "mps"
print(f"Using device: {DEVICE}")

tokenizer = CLIPTokenizer("../scratch/data/vocab.json", merges_file="../scratch/data/merges.txt")
model_file = "../scratch/data/models--runwayml--stable-diffusion-v1-5/snapshots/1d0c4ebf6ff58a5caecab40fa1406526bca4b5b9/v1-5-pruned-emaonly.ckpt"
models = model_loader.preload_models_from_standard_weights(model_file, DEVICE)

# Load experiments from JSON
with open("experiments.json", "r") as f:
    experiments = json.load(f)

# Function to generate images for prompts
def generate_images(category, subcategory, image_path, prompts):
    # Create folder path if missing
    folder_path = Path(f"./generated_images/{category}/{subcategory}/")
    folder_path.mkdir(parents=True, exist_ok=True)
    
    for image_name in os.listdir(image_path):
        if image_name.endswith(".jpg"):
            input_image = Image.open(os.path.join(image_path, image_name))
            for prompt in prompts:
                output_image = None

                # TEXT TO IMAGE
                uncond_prompt = ""  # Also known as negative prompt
                do_cfg = True
                cfg_scale = 8  # min: 1, max: 14

                ## IMAGE TO IMAGE
                strength = 0.9

                ## SAMPLER
                sampler = "ddpm"
                num_inference_steps = 50
                seed = 51828

                output_image = pipeline.generate(
                    prompt=prompt,
                    uncond_prompt=uncond_prompt,
                    input_image=input_image,
                    strength=strength,
                    do_cfg=do_cfg,
                    cfg_scale=cfg_scale,
                    sampler_name=sampler,
                    n_inference_steps=num_inference_steps,
                    seed=seed,
                    models=models,
                    device=DEVICE,
                    idle_device="cpu",
                    tokenizer=tokenizer,
                )

                # Save generated image
                output_image_folder = Path(f"{folder_path}/{prompt}/")
                output_image_folder.mkdir(parents=True, exist_ok=True)
                output_image_name = f"{output_image_folder}/{image_name}"
                output_image_pil = Image.fromarray(output_image)
                output_image_pil = output_image_pil.resize((256, 256))
                output_image_pil.save(output_image_name)
                print(f"Generated and saved image for prompt: {prompt} in {output_image_name}")

# Iterate over experiments
for experiment in experiments["experiments"]:
    category = experiment["category"]
    for subcategory_data in experiment["subcategories"]:
        subcategory = subcategory_data["subcategory"]
        prompts = subcategory_data["prompts"]
        image_path = f"./DATA-1/{category}/{subcategory}/"
        generate_images(category, subcategory, image_path, prompts)
