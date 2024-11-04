# LCM

Latent Consistency Models: Synthesizing High-Resolution Images with Few-step Inference

[![Replicate](https://replicate.com/jyoung105/lcm/badge)](https://replicate.com/jyoung105/lcm/)

## Reference

[![Project](https://img.shields.io/badge/Project-8A2BE2)](https://latent-consistency-models.github.io/)
[![arXiv](https://img.shields.io/badge/arXiv-2310.04378-b31b1b.svg)](https://arxiv.org/abs/2310.04378)
[![GitHub](https://img.shields.io/badge/GitHub-Repo-181717?logo=github)](https://github.com/luosiallen/latent-consistency-model)
[![Hugging Face](https://img.shields.io/badge/🤗%20Huggingface-Model-yellow)](https://huggingface.co/SimianLuo/LCM_Dreamshaper_v7)

## Try

1. clone repo
```
git clone https://github.com/jyoung105/cog-diffusers
```

2. move to directory
```
cd ./cog-diffusers/Consistency/LCM
```

3. download weights before deployment
```
cog run script/download-weights
```

4. predict to inference
```
cog predict -i prompt="an illustration of a man with hoodie on"
```