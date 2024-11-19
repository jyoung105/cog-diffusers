# Hyper-SD

Hyper-SD: Trajectory Segmented Consistency Model for Efficient Image Synthesis

[![Replicate](https://replicate.com/jyoung105/hyper-sdxl/badge)](https://replicate.com/jyoung105/hyper-sdxl/)

## Reference

[![Project](https://img.shields.io/badge/project-8A2BE2)](https://hyper-sd.github.io/)
[![arXiv](https://img.shields.io/badge/arXiv-2404.13686-b31b1b.svg)](https://arxiv.org/abs/2404.13686)
[![Hugging Face](https://img.shields.io/badge/🤗%20Huggingface-Model-yellow)](https://huggingface.co/ByteDance/Hyper-SD)

## Example

1. A bustling Moroccan marketplace at sunset, with vibrant stalls displaying colorful textiles, spices, and lanterns, as merchants and shoppers engage in lively exchanges.
![Alt](../../assets/hyper1.png)

2. An underwater scene featuring a sunken pirate ship surrounded by coral reefs, schools of tropical fish, and a curious sea turtle exploring the wreckage.
![Alt](../../assets/hyper2.png)

3. A steampunk-inspired airship soaring above a Victorian-era city, with intricate gears and steam engines visible, and a crew of adventurers on deck.
![Alt](../../assets/hyper3.png)

4. A tranquil Scandinavian village during winter, with snow-covered rooftops, smoke rising from chimneys, and the Northern Lights illuminating the night sky.
![Alt](../../assets/hyper4.png)

5. A mystical desert landscape with towering sand dunes, an ancient, weathered statue half-buried in the sand, and a lone traveler approaching on camelback under a star-filled sky.
![Alt](../../assets/hyper5.png)

## Abstract

(Summarized by GPT-4o)

The paper titled "Hyper-SD: Trajectory Segmented Consistency Model for Efficient Image Synthesis" introduces Hyper-SD, a framework designed to enhance the efficiency of image generation in diffusion models. Traditional diffusion models often require multiple inference steps, leading to significant computational demands. To address this, Hyper-SD combines two primary approaches: preserving the original Ordinary Differential Equation (ODE) trajectory and reformulating it for improved performance.

**Key components of Hyper-SD include:**

* **Trajectory Segmented Consistency Distillation (TSCD):** This technique divides the time steps into segments, enforcing consistency within each segment. By progressively reducing the number of segments, the model achieves all-time consistency, effectively preserving the original ODE trajectory from a higher-order perspective.

* **Human Feedback Learning:** Incorporating human feedback enhances the model's performance in low-step regimes, mitigating performance loss during the distillation process.

* **Score Distillation with Unified LoRA:** Integrating score distillation further improves low-step generation capabilities. The use of a unified Low-Rank Adaptation (LoRA) supports the inference process across all steps.

Extensive experiments and user studies demonstrate that Hyper-SD achieves state-of-the-art performance with 1 to 8 inference steps for both SDXL and SD1.5 models. For instance, in a 1-step inference scenario, Hyper-SDXL surpasses SDXL-Lightning by +0.68 in CLIP Score and +0.51 in Aesthetic Score.

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
cd ./cog-diffusers/Consistency/Hyper
```

3. download weights before deployment
```
cog run scripts/download-weights
```

4. save pipeline before deployment
```
cog run scripts/save-weights
```

5. predict to inference
```
cog predict -i prompt="an illustration of a man with hoodie on"
```