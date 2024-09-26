# Lightning

SDXL-Lightning: Progressive Adversarial Diffusion Distillation

## Reference

- [arxiv](https://arxiv.org/abs/2402.13929)
- [hugging face](https://huggingface.co/ByteDance/SDXL-Lightning)

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