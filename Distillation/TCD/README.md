# TCD

Trajectory Consistency Distillation: Improved Latent Consistency Distillation by Semi-Linear Consistency Function with Trajectory Mapping

## Reference

- [project](https://mhh0318.github.io/tcd/)
- [arxiv](https://arxiv.org/abs/2402.19159)
- [github](https://github.com/jabir-zheng/TCD)
- [hugging face](https://huggingface.co/h1t/TCD-SDXL-LoRA)

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