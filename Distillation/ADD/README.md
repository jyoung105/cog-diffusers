# ADD

Adversarial Diffusion Distillation

[![Replicate](https://replicate.com/jyoung105/sdxl-turbo/badge)](https://replicate.com/jyoung105/sdxl-turbo/)

## Reference

[![Project](https://img.shields.io/badge/Project-8A2BE2)](https://stability.ai/news/stability-ai-sdxl-turbo)
[![arXiv](https://img.shields.io/badge/arXiv-2311.17042-b31b1b.svg)](https://arxiv.org/pdf/2311.17042)
[![Hugging Face - SDXL](https://img.shields.io/badge/🤗%20Huggingface-Model-yellow)](https://huggingface.co/stabilityai/sdxl-turbo)
[![Hugging Face - SD15](https://img.shields.io/badge/🤗%20Huggingface-Model-yellow)](https://huggingface.co/stabilityai/sd-turbo)

## Try

1. clone repo
```
git clone https://github.com/jyoung105/cog-diffusers
```

2. move to directory
```
cd ./cog-diffusers/Distillation/ADD
```

3. download weights before deployment
```
cog run script/download-weights
```

4. predict to inference
```
cog predict -i prompt="an illustration of a man with hoodie on"
```