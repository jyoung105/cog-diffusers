# Hyper-SD

Hyper-SD: Trajectory Segmented Consistency Model for Efficient Image Synthesis

[![Replicate](https://replicate.com/jyoung105/hyper-sdxl/badge)](https://replicate.com/jyoung105/hyper-sdxl/)

## Reference

[![Project](https://img.shields.io/badge/project-8A2BE2)](https://hyper-sd.github.io/)
[![arXiv](https://img.shields.io/badge/arXiv-2404.13686-b31b1b.svg)](https://arxiv.org/abs/2404.13686)
[![Hugging Face](https://img.shields.io/badge/🤗%20Huggingface-Model-yellow)](https://huggingface.co/ByteDance/Hyper-SD)

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
cog run script/download-weights
```

4. predict to inference
```
cog predict -i prompt="an illustration of a man with hoodie on"
```