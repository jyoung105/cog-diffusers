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

from diffusers import AuraFlowPipeline
from diffusers.utils import logging
from diffusers.utils.logging import set_verbosity

# Cog imports
from cog import BasePredictor, Input, Path

# Set logging level
set_verbosity(logging.ERROR)

# GPU global variables
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32  # fp16 or fp32

# AI global variables
TOTAL_CACHE = "./cache"
REPO_CACHE = "./repo-cache"
MODEL_CACHE = "./auraflow-cache"

MODEL_ID = "fal/AuraFlow"


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
        self.load_aura()


    def load_aura(self) -> None:
        """Load the AuraFlow pipeline and related components."""
        print("[~] Setting up pipeline...")
        
        # load the pipeline
        self.pipe = AuraFlowPipeline.from_pretrained(
            MODEL_CACHE,
            variant="fp16",
            torch_dtype=DTYPE,
            use_safetensors=True,
        )
        
        self.pipe.to(DEVICE, DTYPE)


    @torch.inference_mode()
    def generate_image(
        self,
        prompt: str,
        width: int,
        height: int,
        num_outputs: int,
        num_steps: int,
        guidance_scale: float,
        seed: int,
    ) -> List[Image.Image]:
        """Generate images based on the given prompts and parameters."""
        flush()
        setup_seed(seed)
        print(f"[Debug] Prompt: {prompt}")
        print(f"[Debug] Seed: {seed}")
        
        image_list = self.pipe(
            prompt                = prompt,
            guidance_scale        = guidance_scale,
            num_images_per_prompt = num_outputs,
            num_inference_steps   = num_steps,
            width                 = width,
            height                = height,
        ).images
        
        return image_list


    @torch.inference_mode()
    def predict(
        self,
        prompt: str = Input(
            description="Input prompt, text of what you want to generate.",
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
        steps: int = Input(
            description="Number of denoising steps.",
            default=50,
            ge=1,
            le=50,
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance.",
            default=3.5,
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

        new_prompt = prompt + ", best quality, high detail, sharp focus"
            
        print(f"Setup completed in {time.time() - start_time:.2f} seconds.")
        print("[~] Generating images...")
        generation_start_time = time.time()
            
        images = self.generate_image(
            prompt         = new_prompt,
            width          = width,
            height         = height,
            num_outputs    = num_images,
            num_steps      = steps,
            guidance_scale = guidance_scale,
            seed           = seed,
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