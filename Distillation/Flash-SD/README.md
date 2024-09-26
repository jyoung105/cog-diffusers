# Flash-SD

Flash Diffusion: Accelerating Any Conditional Diffusion Model for Few Steps Image Generation

## Reference

- [project](https://gojasper.github.io/flash-diffusion-project/)
- [arxiv](https://arxiv.org/abs/2406.02347)
- [github](https://github.com/gojasper/flash-diffusion)
- [hugging face-sd15](https://huggingface.co/jasperai/flash-sd)
- [hugging face-sdxl](https://huggingface.co/jasperai/flash-sdxl)
- [hugging face-sd3](https://huggingface.co/jasperai/flash-sd3)
- [hugging face-pixart](https://huggingface.co/jasperai/flash-pixart)

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