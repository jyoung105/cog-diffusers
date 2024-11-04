# DMD2

Improved Distribution Matching Distillation for Fast Image Synthesis

[![Replicate](https://replicate.com/jyoung105/dmd2/badge)](https://replicate.com/jyoung105/dmd2/)

## Reference

[![Project](https://img.shields.io/badge/Project-8A2BE2)](https://tianweiy.github.io/dmd2/)
[![arXiv](https://img.shields.io/badge/arXiv-2405.14867-b31b1b.svg)](https://arxiv.org/abs/2405.14867)
[![GitHub](https://img.shields.io/badge/GitHub-Repo-181717?logo=github)](https://github.com/tianweiy/DMD2)
[![Hugging Face](https://img.shields.io/badge/🤗%20Huggingface-Model-yellow)](https://huggingface.co/tianweiy/DMD2)

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