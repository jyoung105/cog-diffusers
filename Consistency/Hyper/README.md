# Hyper-SD

Hyper-SD: Trajectory Segmented Consistency Model for Efficient Image Synthesis

This one is only working with SDXL. (btw, flux version was released, too.)

## Reference

- [project](https://hyper-sd.github.io/)
- [arxiv](https://arxiv.org/abs/2404.13686)
- [hugging face](https://huggingface.co/ByteDance/Hyper-SD)

## Try

1. clone repo
```
git clone https://github.com/jyoung105/cog-diffusers
```

2. move to directory
```
cd ./cog-diffusers/Consistency/Hyper
```

3. download weights before deployment
```
cog run script/download-weights
```

4. predict to inference
```
cog predict -i prompt="an illustration of a man with hoodie on"
```

### Memo
```
TODO - check why # ! doesn't work while #! works.
#!/usr/bin/env python
```