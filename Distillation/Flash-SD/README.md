# Flash-SD

Flash Diffusion: Accelerating Any Conditional Diffusion Model for Few Steps Image Generation

[![Replicate](https://replicate.com/jyoung105/flash-sdxl/badge)](https://replicate.com/jyoung105/flash-sdxl/)

## Reference

[![Project](https://img.shields.io/badge/Project-8A2BE2)](https://gojasper.github.io/flash-diffusion-project/)
[![arXiv](https://img.shields.io/badge/arXiv-2406.02347-b31b1b.svg)](https://arxiv.org/abs/2406.02347)
[![GitHub](https://img.shields.io/badge/GitHub-Repo-181717?logo=github)](https://github.com/gojasper/flash-diffusion)
[![Hugging Face - SD15](https://img.shields.io/badge/🤗%20Huggingface-Model-yellow)](https://huggingface.co/jasperai/flash-sd)
[![Hugging Face - SDXL](https://img.shields.io/badge/🤗%20Huggingface-Model-yellow)](https://huggingface.co/jasperai/flash-sdxl)
[![Hugging Face - SD3](https://img.shields.io/badge/🤗%20Huggingface-Model-yellow)](https://huggingface.co/jasperai/flash-sd3)
[![Hugging Face - Pixart](https://img.shields.io/badge/🤗%20Huggingface-Model-yellow)](https://huggingface.co/jasperai/flash-pixart)

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
cog run script/download-weights
```

4. predict to inference
```
cog predict -i prompt="an illustration of a man with hoodie on"
```