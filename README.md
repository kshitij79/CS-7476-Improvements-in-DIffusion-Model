# CS-7476-Improvements-in-DIffusion-Model

This repository contains the implementation of advanced techniques in diffusion models, specifically focusing on attention-based inpainting and latent space manipulation. The research and development work here led to the publication titled ["Enhancing Conditional Image Generation with Explainable Latent Space Manipulation"](https://arxiv.org/abs/2408.16232).

## Overview

The primary goal of this project is to enhance text-to-image synthesis by integrating stable diffusion models with novel attention-based techniques. The project employs a dynamic masking technique that utilizes gradient-based selective attention (Grad-SAM) to achieve precise inpainting and improve fidelity preservation.

## Key Features

- **Stable Diffusion Pipeline**: Implementation of the diffusion model pipeline with enhancements for dynamic masking.
- **Gradient-based Selective Attention (Grad-SAM)**: Integration of Grad-SAM for analyzing and manipulating cross attention maps.
- **Latent Space Manipulation**: Techniques for adjusting the latent space to better align generated images with reference features.
- **Performance Metrics**: Evaluation using Frechet Inception Distance (FID) and CLIP scores to assess image quality and textual alignment.

## Acknowledgements

This research was conducted as part of the CS 7476 Advanced Computer Vision course in Spring 2024 under the Professor James Hays.
