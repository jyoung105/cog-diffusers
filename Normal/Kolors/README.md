# Kolors

Kolors: Effective Training of Diffusion Model for Photorealistic Text-to-Image Synthesis

## Reference

- [project](https://kwai-kolors.github.io/)
- [arxiv](https://github.com/Kwai-Kolors/Kolors/blob/master/imgs/Kolors_paper.pdf)
- [github](https://github.com/Kwai-Kolors/Kolors)
- [hugging face](https://huggingface.co/Kwai-Kolors/Kolors)
- [hugging face-diffusers](https://huggingface.co/Kwai-Kolors/Kolors-diffusers)

## Try

1. clone repo
```
git clone https://github.com/jyoung105/cog-diffusers
```

2. move to directory
```
cd ./cog-diffusers/Normal/Kolors
```

3. download weights before deployment
```
cog run script/download-weights
```

4. predict to inference
```
cog predict -i prompt="an illustration of a man with hoodie on"
```