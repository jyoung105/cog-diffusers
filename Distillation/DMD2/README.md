# DMD2

Improved Distribution Matching Distillation for Fast Image Synthesis

## Reference

- [project](https://tianweiy.github.io/dmd2/)
- [arxiv](https://arxiv.org/abs/2405.14867)
- [github](https://github.com/tianweiy/DMD2)
- [hugging face](https://huggingface.co/tianweiy/DMD2)

## Try

1. clone repo
```
git clone https://github.com/jyoung105/cog-diffusers
```

2. move to directory
```
cd ./cog-diffusers/Distillation/DMD2
```

3. download weights before deployment
```
cog run script/download-weights
```

4. predict to inference
```
cog predict -i prompt="an illustration of a man with hoodie on"
```