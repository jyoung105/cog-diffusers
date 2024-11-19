# TCD

Trajectory Consistency Distillation: Improved Latent Consistency Distillation by Semi-Linear Consistency Function with Trajectory Mapping

[![Replicate](https://replicate.com/jyoung105/tcd-sdxl/badge)](https://replicate.com/jyoung105/tcd-sdxl/)

## Reference

[![Project](https://img.shields.io/badge/Project-8A2BE2)](https://mhh0318.github.io/tcd/)
[![arXiv](https://img.shields.io/badge/arXiv-2402.19159-b31b1b.svg)](https://arxiv.org/abs/2402.19159)
[![GitHub](https://img.shields.io/badge/GitHub-Repo-181717?logo=github)](https://github.com/jabir-zheng/TCD)
[![Hugging Face](https://img.shields.io/badge/🤗%20Huggingface-Model-yellow)](https://huggingface.co/h1t/TCD-SDXL-LoRA)

## Example

1. A bustling Moroccan marketplace at sunset, with vibrant stalls displaying colorful textiles, spices, and lanterns, as merchants and shoppers engage in lively exchanges.
![Alt](../../assets/tcd1.png)

2. An underwater scene featuring a sunken pirate ship surrounded by coral reefs, schools of tropical fish, and a curious sea turtle exploring the wreckage.
![Alt](../../assets/tcd2.png)

3. A steampunk-inspired airship soaring above a Victorian-era city, with intricate gears and steam engines visible, and a crew of adventurers on deck.
![Alt](../../assets/tcd3.png)

4. A tranquil Scandinavian village during winter, with snow-covered rooftops, smoke rising from chimneys, and the Northern Lights illuminating the night sky.
![Alt](../../assets/tcd4.png)

5. A mystical desert landscape with towering sand dunes, an ancient, weathered statue half-buried in the sand, and a lone traveler approaching on camelback under a star-filled sky.
![Alt](../../assets/tcd5.png)

## Abstract

(Summarized by GPT-4o)

The paper titled "Trajectory Consistency Distillation: Improved Latent Consistency Distillation by Semi-Linear Consistency Function with Trajectory Mapping" introduces Trajectory Consistency Distillation (TCD), a method aimed at enhancing the efficiency and quality of text-to-image synthesis in Latent Consistency Models (LCMs).

**Key Contributions:**

* **Trajectory Consistency Function:** TCD broadens the self-consistency boundary condition through trajectory mapping, enabling accurate tracing of the entire trajectory of the Probability Flow Ordinary Differential Equation (ODE) in a semi-linear form using an Exponential Integrator. This approach reduces parameterization and distillation errors, leading to improved image clarity and detail. 

* **Strategic Stochastic Sampling:** The method introduces explicit control over stochasticity, mitigating accumulated errors inherent in multi-step consistency sampling. This results in more precise and reliable image generation. 

* **Enhanced Image Quality:** Experimental results demonstrate that TCD significantly improves image quality at low Numbers of Function Evaluations (NFEs) and produces more detailed results compared to the teacher model at high NFEs. 

The authors have provided additional resources, including a project page, to facilitate further exploration and application of TCD.

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
cd ./cog-diffusers/Distillation/TCD
```

3. download weights before deployment
```
cog run scripts/download-weights
```

4. predict to inference
```
cog predict -i prompt="an illustration of a man with hoodie on"
```