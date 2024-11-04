# PCM

Phased Consistency Model

[![Replicate](https://replicate.com/jyoung105/pcm/badge)](https://replicate.com/jyoung105/pcm/)

## Reference

[![Project](https://img.shields.io/badge/Project-8A2BE2)](https://g-u-n.github.io/projects/pcm/)
[![arXiv](https://img.shields.io/badge/arXiv-2405.18407-b31b1b.svg)](https://arxiv.org/abs/2405.18407)
[![GitHub](https://img.shields.io/badge/GitHub-Repo-181717?logo=github)](https://github.com/G-U-N/Phased-Consistency-Model)
[![Hugging Face](https://img.shields.io/badge/🤗%20Huggingface-Model-yellow)](https://huggingface.co/wangfuyun/PCM_Weights)

## Try

1. clone repo
```
git clone https://github.com/jyoung105/cog-diffusers
```

2. move to directory
```
cd ./cog-diffusers/Consistency/PCM
```

3. download weights before deployment
```
cog run script/download-weights
```

4. predict to inference
```
cog predict -i prompt="an illustration of a man with hoodie on"
```