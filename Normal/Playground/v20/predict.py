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

from diffusers import StableDiffusionXLPipeline
from diffusers.utils import logging
from diffusers.utils.logging import set_verbosity

from safetensors.torch import load_file

from compel import (
    Compel, 
    ReturnedEmbeddingsType, 
    DiffusersTextualInversionManager,
)

# Cog imports
from cog import BasePredictor, Input, Path

# Set logging level
set_verbosity(logging.ERROR)

# GPU global variables
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32  # fp16 or fp32

# AI global variables
TOTAL_CACHE = "./cache"
MODEL_CACHE = "./model-cache"

MODEL_ID = "playgroundai/playground-v2-1024px-aesthetic"
MODEL_FILE = "playground-v2.fp16.safetensors"


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
        self.load_playground()


    def load_playground(self) -> None:
        """Load the Stable Diffusion XL pipeline and related components."""
        print("[~] Setting up pipeline...")
        
        # load the pipeline
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            MODEL_CACHE,
            variant="fp16",
            torch_dtype=DTYPE,
            use_safetensors=True,
        )
        
        self.pipe.to(DEVICE, DTYPE)
        
        # Enable optimizations
        # Inference speed
        self.pipe.enable_vae_slicing()
        self.pipe.enable_vae_tiling()
        self.pipe.enable_attention_slicing()
        
        # Memory
        # self.pipe.enable_model_cpu_offload() # This optimization slightly reduces memory consumption, but is optimized for speed.
        # self.pipe.enable_sequential_cpu_offload() # This optimization reduces memory consumption, but also reduces speed.
        # self.pipe.enable_xformers_memory_efficient_attention() # useless for torch > 2.0, but if using torch < 2.0, this is an essential optimization.
        
        # PyTorch
        # self.pipe.unet.set_attn_processor(AttnProcessor2_0())
        self.pipe.fuse_qkv_projections()
        self.pipe.unet.to(memory_format=torch.channels_last)
        self.pipe.vae.to(memory_format=torch.channels_last)
        
        # self.pipe.unet = torch.compile(self.pipe.unet, mode="reduce-overhead", fullgraph=True) # max-autotune or reduce-overhead
        # self.pipe.vae.decode = torch.compile(self.pipe.vae.decode, mode="reduce-overhead", fullgraph=True)
        self.pipe.upcast_vae()
        
        # Setup Compel
        self.textual_inversion_manager = DiffusersTextualInversionManager(self.pipe)
        self.compel = Compel(
            tokenizer=[self.pipe.tokenizer, self.pipe.tokenizer_2],
            text_encoder=[self.pipe.text_encoder, self.pipe.text_encoder_2],
            textual_inversion_manager=self.textual_inversion_manager,
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED, 
            requires_pooled=[False, True]
        )


    @torch.inference_mode()
    def generate_image(
        self,
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        num_outputs: int,
        num_steps: int,
        eta: float,
        guidance_scale: float,
        seed: int,
        clip_skip: int,
    ) -> List[Image.Image]:
        """Generate images based on the given prompts and parameters."""
        flush()
        setup_seed(seed)
        print(f"[Debug] Prompt: {prompt}")
        print(f"[Debug] Seed: {seed}")
        
        # Convert prompt, negative_prompt to embeddings
        conditioning, pooled = self.compel(prompt)
        # neg_conditioning, neg_pooled = self.compel(negative_prompt) # error when we use more than 2 embeddings
        
        image_list = self.pipe(
            prompt_embeds         = conditioning,
            pooled_prompt_embeds  = pooled,
            negative_prompt       = negative_prompt,
            eta                   = eta,
            guidance_scale        = guidance_scale,
            num_images_per_prompt = num_outputs,
            num_inference_steps   = num_steps,
            width                 = width,
            height                = height,
            clip_skip             = clip_skip,
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
        steps: int = Input(
            description="Number of denoising steps.",
            default=25,
            ge=1,
            le=50,
        ),
        eta: float = Input(
            description="Stochastic parameter to control the randomness.",
            default=0,
            ge=0,
            le=1,
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance.",
            default=3.0,
            ge=0,
            le=20,
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed.",
            default=None,
        ),
        clip_skip: int = Input(
            description="Number of the layers to skip in CLIP.",
            default=0,
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
            prompt          = new_prompt,
            negative_prompt = new_negative_prompt,
            width           = width,
            height          = height,
            num_outputs     = num_images,
            num_steps       = steps,
            eta             = eta,
            guidance_scale  = guidance_scale,
            seed            = seed,
            clip_skip       = clip_skip
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