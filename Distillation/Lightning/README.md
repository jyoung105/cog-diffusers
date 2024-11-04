# Lightning

SDXL-Lightning: Progressive Adversarial Diffusion Distillation

[![Replicate](https://replicate.com/jyoung105/lightning-turbo/badge)](https://replicate.com/jyoung105/lightning-turbo/)

## Reference

[![arXiv](https://img.shields.io/badge/arXiv-2402.13929-b31b1b.svg)](https://arxiv.org/abs/2402.13929)
[![Hugging Face](https://img.shields.io/badge/🤗%20Huggingface-Model-yellow)](https://huggingface.co/ByteDance/SDXL-Lightning)

## Try

1. clone repo
```
git clone https://github.com/jyoung105/cog-diffusers
```

2. move to directory
```
cd ./cog-diffusers/Distillation/Lightning
```

3. download weights before deployment
```
cog run script/download-weights
```

4. predict to inference
```
cog predict -i prompt="an illustration of a man with hoodie on"
```