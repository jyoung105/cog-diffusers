# PCM

Phased Consistency Model

## Reference

- [project](https://g-u-n.github.io/projects/pcm/)
- [arxiv](https://arxiv.org/abs/2405.18407)
- [github](https://github.com/G-U-N/Phased-Consistency-Model)
- [hugging face](https://huggingface.co/wangfuyun/PCM_Weights)

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

## Memo
- check the quality issue