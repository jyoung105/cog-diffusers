# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

# Standard library imports
import os
import random
import gc
import time
from typing import List

# Third-party library imports
import torch
import numpy as np
from PIL import Image

from diffusers import StableCascadeDecoderPipeline, StableCascadePriorPipeline
from diffusers.utils import logging
from diffusers.utils.logging import set_verbosity

from safetensors.torch import load_file

# Cog imports
from cog import BasePredictor, Input, Path

# Set logging level
set_verbosity(logging.ERROR)

# GPU global variables
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32  # bf16 or fp32

# AI global variables
PRIOR_CACHE = "./prior-cache"
DECODER_CACHE = "./decoder-cache"
PRIOR_REPO_CACHE = "./prior-repo-cache"
DECODER_REPO_CACHE = "./decoder-repo-cache"

PRIOR_ID = "stabilityai/stable-cascade-prior"
DECODER_ID = "stabilityai/stable-cascade"


# Set safety checker
# SAFETY_CACHE = "./safetys"
# FEATURE_EXTRACTOR = "./feature-extractors"
# SAFETY_URL = "https://weights.replicate.delivery/default/playgroundai/safety-cache.tar"


def setup_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)                          # Set the Python built-in random seed
    np.random.seed(seed)                       # Set the NumPy random seed
    torch.manual_seed(seed)                    # Set the PyTorch random seed for CPU
    torch.cuda.manual_seed_all(seed)           # Set the PyTorch random seed for all GPUs
    torch.backends.cudnn.benchmark = False     # Disable CUDNN benchmark for deterministic behavior
    torch.backends.cudnn.deterministic = True  # Ensure deterministic CUDNN operations


def flush() -> None:
    """Clear GPU cache."""
    torch.cuda.synchronize() # Synchronize CUDA operations
    gc.collect()             # Collect garbage
    torch.cuda.empty_cache() # Empty CUDA cache


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient."""
        self.load_cascade()


    def load_cascade(self) -> None:
        """Load the Stable Cascade pipeline and related components."""
        print("[~] Setting up pipeline...")
        
        # load the pipeline
        self.prior = StableCascadePriorPipeline.from_pretrained(
            PRIOR_CACHE,
            variant="bf16",
            torch_dtype=DTYPE,
            use_safetensors=True,
        )
        
        self.prior.to(DEVICE, DTYPE)
        
        self.decoder = StableCascadeDecoderPipeline.from_pretrained(
            DECODER_CACHE,
            variant="bf16",
            torch_dtype=DTYPE,
            use_safetensors=True,
        )
        
        self.decoder.to(DEVICE, DTYPE)
        
        # Enable optimizations
        # Memory
        self.prior.enable_model_cpu_offload()
        self.decoder.enable_model_cpu_offload()


    @torch.inference_mode()
    def generate_image(
        self,
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        num_outputs: int,
        num_steps_prior: int,
        num_steps_decoder: int,
        guidance_scale_prior: float,
        guidance_scale_decoder: float,
        seed: int,
    ) -> List[Image.Image]:
        """Generate images based on the given prompts and parameters."""
        flush()
        setup_seed(seed)
        print(f"[Debug] Prompt: {prompt}")
        print(f"[Debug] Seed: {seed}")
        
        prior_list = self.prior(
            prompt                = prompt,
            negative_prompt       = negative_prompt,
            guidance_scale        = guidance_scale_prior,
            num_images_per_prompt = num_outputs,
            num_inference_steps   = num_steps_prior,
            width                 = width,
            height                = height,
        )
        
        image_list = self.decoder(
            image_embeddings    = prior_list.image_embeddings,
            prompt              = prompt,
            negative_prompt     = negative_prompt,
            guidance_scale      = guidance_scale_decoder,
            num_inference_steps = num_steps_decoder,
            output_type         = "pil",
        ).images
        
        return image_list


    @torch.inference_mode()
    def predict(
        self,
        prompt: str = Input(
            description="Input prompt, text of what you want to generate.",
            default=None,
        ),
        negative_prompt: str = Input(
            description="Input negative prompt, text of what you don't want to generate.",
            default=None,
        ),
        width: int = Input(
            description="Width of the output image.",
            default=1024,
            ge=1,
            le=2048,
        ),
        height: int = Input(
            description="Height of the output image.",
            default=1024,
            ge=1,
            le=2048,
        ),
        num_images: int = Input(
            description="Number of output images.",
            default=1,
            ge=1,
            le=4,
        ),
        steps_prior: int = Input(
            description="Number of denoising steps in prior.",
            default=20,
            ge=1,
            le=50,
        ),
        steps_decoder: int = Input(
            description="Number of denoising steps in decoder.",
            default=10,
            ge=1,
            le=50,
        ),
        guidance_scale_prior: float = Input(
            description="Scale for classifier-free guidance in prior.",
            default=4.0,
            ge=0,
            le=20,
        ),
        guidance_scale_decoder: float = Input(
            description="Scale for classifier-free guidance in decoder.",
            default=0.0,
            ge=0,
            le=20,
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed.",
            default=None,
        ),
        # output_type: str = Input(
        #     description="Format of the output",
        #     default="webp",
        #     choices=["png", "jpg", "webp"]
        # ),
        # output_quality: int = Input(
        #     description="Quality of the output",
        #     default=80,
        #     ge=0,
        #     le=100,
        # ),
    ) -> List[Path]:
        """Run a prediction to generate images based on the input parameters."""
        start_time = time.time()
        
        if not prompt:
            print("No input prompt provided.")
            return []

        print(f"[Debug] DEVICE: {DEVICE}")
        print(f"[Debug] DTYPE: {DTYPE}")
        
        # If no seed is provided, generate a random seed
        if seed is None:
            seed = random.randint(0, 65535)

        # Set prompt and negative_prompt
        negative_prompt = negative_prompt or ""

        new_prompt = f"{prompt}, best quality, high detail, sharp focus"
        new_negative_prompt = f"{negative_prompt}"
        
        print(f"Setup completed in {time.time() - start_time:.2f} seconds.")
        print("[~] Generating images...")
        generation_start_time = time.time()
            
        images = self.generate_image(
            prompt                 = new_prompt,
            negative_prompt        = new_negative_prompt,
            width                  = width,
            height                 = height,
            num_outputs            = num_images,
            num_steps_prior        = steps_prior,
            num_steps_decoder      = steps_decoder,
            guidance_scale_prior   = guidance_scale_prior,
            guidance_scale_decoder = guidance_scale_decoder,
            seed                   = seed,
        )
        print(f"Image generation completed in {time.time() - generation_start_time:.2f} seconds.")

        # Save the generated images
        # TODO : Check for NSFW content
        output_paths = []
        for i, image in enumerate(images):
            output_path = f"/tmp/out_{i}.png"
            image.save(output_path)
            output_paths.append(Path(output_path))
        
        return output_paths