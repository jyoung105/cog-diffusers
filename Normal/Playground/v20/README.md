# Playground-v2.0

Playground v2.0: A diffusion-based text-to-image generation model trained from scratch by the research team at Playground

## Reference

- [project](https://playground.com/blog/playground-v2)
- [hugging face](playgroundai/playground-v2-1024px-aesthetic)

## Try

1. clone repo
```
git clone https://github.com/jyoung105/cog-diffusers
```

2. move to directory
```
cd ./cog-diffusers/Normal/Playground/v20
```

3. download weights before deployment
```
cog run script/download-weights
```

4. predict to inference
```
cog predict -i prompt="an illustration of a man with hoodie on"
```