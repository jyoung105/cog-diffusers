# ADD

Adversarial Diffusion Distillation

## Reference

- [project](https://stability.ai/news/stability-ai-sdxl-turbo)
- [arxiv](https://arxiv.org/pdf/2311.17042)
- [hugging face 1](https://huggingface.co/stabilityai/sdxl-turbo)
- [hugging face 2](https://huggingface.co/stabilityai/sd-turbo)

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