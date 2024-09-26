# Playground-v2.5

Playground v2.5: Three Insights towards Enhancing Aesthetic Quality in Text-to-Image Generation

## Reference

- [project](https://playground.com/blog/playground-v2-5)
- [arxiv](https://arxiv.org/html/2402.17245v1)
- [hugging face](playgroundai/playground-v2.5-1024px-aesthetic)

## Try

1. clone repo
```
git clone https://github.com/jyoung105/cog-diffusers
```

2. move to directory
```
cd ./cog-diffusers/Normal/Playground/v25
```

3. download weights before deployment
```
cog run script/download-weights
```

4. predict to inference
```
cog predict -i prompt="an illustration of a man with hoodie on"
```