# Flash-SD

Flash Diffusion: Accelerating Any Conditional Diffusion Model for Few Steps Image Generation

[![Replicate](https://replicate.com/jyoung105/flash-sdxl/badge)](https://replicate.com/jyoung105/flash-sdxl/)

## Reference

[![Project](https://img.shields.io/badge/Project-8A2BE2)](https://gojasper.github.io/flash-diffusion-project/)
[![arXiv](https://img.shields.io/badge/arXiv-2406.02347-b31b1b.svg)](https://arxiv.org/abs/2406.02347)
[![GitHub](https://img.shields.io/badge/GitHub-Repo-181717?logo=github)](https://github.com/gojasper/flash-diffusion)
[![Hugging Face - SD15](https://img.shields.io/badge/🤗%20Huggingface-Model_1-yellow)](https://huggingface.co/jasperai/flash-sd)
[![Hugging Face - SDXL](https://img.shields.io/badge/🤗%20Huggingface-Model_2-yellow)](https://huggingface.co/jasperai/flash-sdxl)
[![Hugging Face - SD3](https://img.shields.io/badge/🤗%20Huggingface-Model_3-yellow)](https://huggingface.co/jasperai/flash-sd3)
[![Hugging Face - Pixart](https://img.shields.io/badge/🤗%20Huggingface-Model_4-yellow)](https://huggingface.co/jasperai/flash-pixart)

## Example

1. A bustling Moroccan marketplace at sunset, with vibrant stalls displaying colorful textiles, spices, and lanterns, as merchants and shoppers engage in lively exchanges.
![Alt](../../assets/flash1.png)

2. An underwater scene featuring a sunken pirate ship surrounded by coral reefs, schools of tropical fish, and a curious sea turtle exploring the wreckage.
![Alt](../../assets/flash2.png)

3. A steampunk-inspired airship soaring above a Victorian-era city, with intricate gears and steam engines visible, and a crew of adventurers on deck.
![Alt](../../assets/flash3.png)

4. A tranquil Scandinavian village during winter, with snow-covered rooftops, smoke rising from chimneys, and the Northern Lights illuminating the night sky.
![Alt](../../assets/flash4.png)

5. A mystical desert landscape with towering sand dunes, an ancient, weathered statue half-buried in the sand, and a lone traveler approaching on camelback under a star-filled sky.
![Alt](../../assets/flash5.png)

## Abstract

(Summarized by GPT-4o)

The paper titled "Flash Diffusion: Accelerating Any Conditional Diffusion Model for Few Steps Image Generation" introduces Flash Diffusion, an efficient and versatile distillation method designed to accelerate the generation process of pre-trained diffusion models. 

**Key Contributions:**

* **Efficient Distillation Method:** Flash Diffusion enables rapid image generation by significantly reducing the number of sampling steps required, achieving state-of-the-art performance in terms of Fréchet Inception Distance (FID) and CLIP-Score on datasets like COCO2014 and COCO2017. Notably, this method requires only several GPU hours of training and utilizes fewer trainable parameters compared to existing approaches. 

* **Versatility Across Tasks and Models:** The method demonstrates adaptability across various tasks, including text-to-image generation, inpainting, face-swapping, and super-resolution. It is compatible with different backbones such as UNet-based denoisers (e.g., SD1.5, SDXL) and DiT (e.g., Pixart-α), as well as adapters. In all cases, Flash Diffusion effectively reduces the number of sampling steps while maintaining high-quality image generation. 

The authors have made the official implementation of Flash Diffusion publicly available, facilitating further research and application in the field.

## TODO
- [x] Inference code
- [ ] Method overview
- [ ] Train code

## Try

1. clone repo
```
git clone https://github.com/jyoung105/cog-diffusers
```

2. move to directory
```
cd ./cog-diffusers/Distillation/Flash-SD
```

3. download weights before deployment
```
cog run scripts/download-weights
```

4. predict to inference
```
cog predict -i prompt="an illustration of a man with hoodie on"
```