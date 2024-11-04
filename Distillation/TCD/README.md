# TCD

Trajectory Consistency Distillation: Improved Latent Consistency Distillation by Semi-Linear Consistency Function with Trajectory Mapping

[![Replicate](https://replicate.com/jyoung105/tcd-sdxl/badge)](https://replicate.com/jyoung105/tcd-sdxl/)

## Reference

[![Project](https://img.shields.io/badge/Project-8A2BE2)](https://mhh0318.github.io/tcd/)
[![arXiv](https://img.shields.io/badge/arXiv-2402.19159-b31b1b.svg)](https://arxiv.org/abs/2402.19159)
[![GitHub](https://img.shields.io/badge/GitHub-Repo-181717?logo=github)](https://github.com/jabir-zheng/TCD)
[![Hugging Face](https://img.shields.io/badge/🤗%20Huggingface-Model-yellow)](https://huggingface.co/h1t/TCD-SDXL-LoRA)

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
cog run script/download-weights
```

4. predict to inference
```
cog predict -i prompt="an illustration of a man with hoodie on"
```