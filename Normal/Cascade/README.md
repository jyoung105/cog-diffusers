# Cascade

Würstchen: An Efficient Architecture for Large-Scale Text-to-Image Diffusion Models

## Reference

- [project](https://stability.ai/news/introducing-stable-cascade)
- [arxiv](https://openreview.net/forum?id=gU58d5QeGv)
- [github](https://github.com/Stability-AI/StableCascade)
- [hugging face](https://huggingface.co/stabilityai/stable-cascade)

## Try

1. clone repo
```
git clone https://github.com/jyoung105/cog-diffusers
```

2. move to directory
```
cd ./cog-diffusers/Normal/Cascade
```

3. download weights before deployment
```
cog run script/download-weights
```

4. predict to inference
```
cog predict -i prompt="an illustration of a man with hoodie on"
```