# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
from typing import List

import os

import gc
import time

import torch
import torch.nn.functional as F
import numpy as np

import cv2
import PIL
from PIL import Image

from diffusers import KolorsPipeline, DPMSolverMultistepScheduler
from diffusers.utils import load_image, logging
from diffusers.utils.logging import set_verbosity

from safetensors.torch import load_file


set_verbosity(logging.ERROR)


# GPU global variables
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if str(DEVICE).__contains__("cuda") else torch.float32


# AI global variables
MODEL_CACHE = "./model-cache"

MODEL_ID = "Kwai-Kolors/Kolors-diffusers"


# Set safety checker
# SAFETY_CACHE = "./safetys"
# FEATURE_EXTRACTOR = "./feature-extractors"
# SAFETY_URL = "https://weights.replicate.delivery/default/playgroundai/safety-cache.tar"


def flush():
    gc.collect()
    torch.cuda.empty_cache()


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.load_kolors()


    def load_kolors(self):
        print("[~] Setup pipeline")
        # 1. Setup pipeline
        # self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        #     "h94/IP-Adapter",
        #     subfolder="models/image_encoder",
        #     torch_dtype=torch.float16,
        # )
        self.pipe = KolorsPipeline.from_pretrained(
            MODEL_CACHE,
            # image_encoder=self.image_encoder,
            variant="fp16",
            torch_dtype=torch.float16,
            use_safetensors=True,
        )
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config,
            use_karras_sigmas=True,
        )

        
        # 2. Setup IP-Adapter 
        # self.pipe.load_ip_adapter(
        #     ["ostris/ip-composition-adapter", "h94/IP-Adapter"],
        #     subfolder=["", "sdxl_models"],
        #     weight_name=[
        #         "ip_plus_composition_sdxl.safetensors",
        #         "ip-adapter_sdxl_vit-h.safetensors",
        #     ],
        #     image_encoder_folder=None,
        # ) 
        # self.pipe.load_ip_adapter(
        #     ["h94/IP-Adapter"],
        #     subfolder=["sdxl_models"],
        #     weight_name=[
        #         "ip-adapter-plus_sdxl_vit-h.safetensors",
        #     ],
        #     image_encoder_folder=None,
        # )
        
        self.pipe = self.pipe.to(DEVICE)
        
        
        # 3. Enable to use FreeU
        # self.pipe.enable_freeu(s1=0.9, s2=0.2, b1=1.1, b2=1.2)
        
        
        # 4. Add textual inversion
        # embedding_1 = load_file(f"{TOTAL_CACHE}/ac_neg1.safetensors")
        # embedding_2 = load_file(f"{TOTAL_CACHE}/ac_neg2.safetensors")
        
        # self.pipe.load_textual_inversion(embedding_1["clip_l"], token="<ac_neg1>", text_encoder=self.pipe.text_encoder, tokenizer=self.pipe.tokenizer)
        # self.pipe.load_textual_inversion(embedding_1["clip_g"], token="<ac_neg1>", text_encoder=self.pipe.text_encoder_2, tokenizer=self.pipe.tokenizer_2)
        # self.pipe.load_textual_inversion(embedding_2["clip_l"], token="<ac_neg2>", text_encoder=self.pipe.text_encoder, tokenizer=self.pipe.tokenizer)
        # self.pipe.load_textual_inversion(embedding_2["clip_g"], token="<ac_neg2>", text_encoder=self.pipe.text_encoder_2, tokenizer=self.pipe.tokenizer_2)
        
        
        # 5. Add LoRA
        # self.pipe.load_lora_weights(hf_hub_download("jyoung105/general-lora", "add-detail-xl.safetensors"), adapter_name="<add_detail>")
        # self.pipe.load_lora_weights(hf_hub_download("jyoung105/general-lora", "sd_xl_offset_example-lora_1.0.safetensors"), adapter_name="<noise_offset>")
        # self.pipe.load_lora_weights(hf_hub_download("jyoung105/general-lora", "xl_more_art-full_v1.safetensors"), adapter_name="<art_full>")
        
        # self.pipe.set_adapters(["<add_detail>", "<noise_offset>", "<art_full>"], adapter_weights=[0.5, 0.5, 0.5])
        
        
        # 6. Save memory and improve speed
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
        
        
        # 7. Setup Compel
        # self.textual_inversion_manager = DiffusersTextualInversionManager(self.pipe)
        # self.compel = Compel(
        #     tokenizer=[self.pipe.tokenizer, self.pipe.tokenizer_2],
        #     text_encoder=[self.pipe.text_encoder, self.pipe.text_encoder_2],
        #     textual_inversion_manager=self.textual_inversion_manager,
        #     returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED, 
        #     requires_pooled=[False, True]
        #     )


    @torch.inference_mode()
    def generate_image(
        self,
        prompt,
        negative_prompt,
        width,
        height,
        num_outputs,
        num_steps,
        eta,
        guidance_scale,
        seed,
    ):
        flush()
        print(f"[Debug] Prompt: {prompt}")
        
        generator = torch.Generator(device=DEVICE).manual_seed(seed)
        
        # Convert prompt, negative_prompt to embeddings
        # conditioning, pooled = self.compel(prompt)
        # neg_conditioning, neg_pooled = self.compel(negative_prompt) # error when we use more than 2 embeddings
        
        image_list = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            eta=eta,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_outputs,
            num_inference_steps=num_steps,
            width=width,
            height=height,
            generator=generator,
        ).images
        
        return image_list


    @torch.inference_mode()
    def predict(
        self,
        prompt: str = Input(
            description="Input prompt, text what you want to put on.",
            default=None,
        ),
        negative_prompt: str = Input(
            description="Input negative prompt, text what you don't want to put on.",
            default=None,
        ),
        width: int = Input(
            description="Width of an output.",
            default=1024,
            ge=1,
            le=2048,
        ),
        height: int = Input(
            description="Height of an output.",
            default=1024,
            ge=1,
            le=2048,
        ),
        num_images: int = Input(
            description="Number of outputs.",
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
            description="A stochastic parameter referred to as 'gamma' used to control the stochasticity in every step.",
            default=0.0,
            ge=0,
            le=1,
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance.",
            default=6.5,
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
        """Run a single prediction on the model"""
        start1 = time.time() # stamp time
        
        if prompt is None:
            msg = "No input, Save money"
            return msg

        else:
            print(f"DEVICE: {DEVICE}")
            print(f"DTYPE: {DTYPE}")
            
            # If no seed is provided, generate a random seed
            if seed is None:
                seed = int.from_bytes(os.urandom(2), "big")
            print(f"Using seed: {seed}")

            # Set prompt and negative_prompt
            if negative_prompt is None:
                negative_prompt = ""

            new_prompt = prompt + ", best quality, high detail, sharp focus"
            new_negative_prompt = negative_prompt # "<ac_neg1>, <ac_neg2>, " + 
            
            print("Finish setup in " + str(time.time()-start1) + " secs.")

            start2 = time.time() # stamp time
            
            base_image = self.generate_image(
                prompt=new_prompt,
                negative_prompt=new_negative_prompt,
                width=width,
                height=height,
                num_outputs=num_images,
                num_steps=steps,
                eta=eta,
                guidance_scale=guidance_scale,
                seed=seed,
            )
            print("Finish generation in " + str(time.time()-start2) + " secs.")

            # Save the generated images and check for NSFW content
            output_paths = []
            for i, image in enumerate(base_image):
                output_path = f"/tmp/out_{i}.png"
                image.save(output_path)
                output_paths.append(Path(output_path))
            
            return output_paths