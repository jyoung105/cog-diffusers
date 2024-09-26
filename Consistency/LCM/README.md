# LCM

Latent Consistency Models: Synthesizing High-Resolution Images with Few-step Inference

## Reference

- [project](https://latent-consistency-models.github.io/)
- [arxiv](https://arxiv.org/abs/2310.04378)
- [github](https://github.com/luosiallen/latent-consistency-model)
- [hugging face](https://huggingface.co/SimianLuo/LCM_Dreamshaper_v7)

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