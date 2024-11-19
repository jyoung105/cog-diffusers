# PeRFlow

PeRFlow: Piecewise Rectified Flow as Universal Plug-and-Play Accelerator

[![Replicate](https://replicate.com/jyoung105/perflow/badge)](https://replicate.com/jyoung105/perflow/)

## Reference

[![Project](https://img.shields.io/badge/Project-8A2BE2)](https://piecewise-rectified-flow.github.io/)
[![arXiv](https://img.shields.io/badge/arXiv-2405.07510-b31b1b.svg)](https://arxiv.org/abs/2405.07510)
[![GitHub](https://img.shields.io/badge/GitHub-Repo-181717?logo=github)](https://github.com/magic-research/piecewise-rectified-flow)
[![Hugging Face - SD15](https://img.shields.io/badge/🤗%20Huggingface-Model_1-yellow)](https://huggingface.co/hansyan/perflow-sd15-dreamshaper)
[![Hugging Face - SDXL](https://img.shields.io/badge/🤗%20Huggingface-Model_2-yellow)](https://huggingface.co/hansyan/perflow-sdxl-base)

## Example

1. A bustling Moroccan marketplace at sunset, with vibrant stalls displaying colorful textiles, spices, and lanterns, as merchants and shoppers engage in lively exchanges.
![Alt](../../assets/pf1.png)

2. An underwater scene featuring a sunken pirate ship surrounded by coral reefs, schools of tropical fish, and a curious sea turtle exploring the wreckage.
![Alt](../../assets/pf2.png)

3. A steampunk-inspired airship soaring above a Victorian-era city, with intricate gears and steam engines visible, and a crew of adventurers on deck.
![Alt](../../assets/pf3.png)

4. A tranquil Scandinavian village during winter, with snow-covered rooftops, smoke rising from chimneys, and the Northern Lights illuminating the night sky.
![Alt](../../assets/pf4.png)

5. A mystical desert landscape with towering sand dunes, an ancient, weathered statue half-buried in the sand, and a lone traveler approaching on camelback under a star-filled sky.
![Alt](../../assets/pf5.png)

## Abstract

(Summarized by GPT-4o)

The paper titled "PeRFlow: Piecewise Rectified Flow as Universal Plug-and-Play Accelerator" introduces **Piecewise Rectified Flow (PeRFlow)**, a method designed to enhance the efficiency of diffusion models in generative tasks. 

**Key Contributions:**

* **Piecewise Linear Flow Approximation:** PeRFlow segments the sampling process of generative flows into multiple time windows. Within each interval, it straightens the trajectories using a reflow operation, effectively approximating piecewise linear flows. This approach enables superior performance in few-step generation scenarios. 

* **Knowledge Inheritance from Pretrained Models:** Through specialized parameterizations, PeRFlow models can inherit knowledge from pretrained diffusion models. This inheritance facilitates rapid convergence during training and enhances the models' transferability across various tasks. 

* **Universal Plug-and-Play Compatibility:** The design of PeRFlow allows it to serve as a universal plug-and-play accelerator, making it compatible with a wide range of workflows that utilize pretrained diffusion models. 

The authors have made the codes for training and inference publicly available, promoting further research and application of PeRFlow in the field of machine learning.  

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
cd ./cog-diffusers/Flow/PeRFlow
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