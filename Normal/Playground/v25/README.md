# Playground-v2.5

Playground v2.5: Three Insights towards Enhancing Aesthetic Quality in Text-to-Image Generation

[![Replicate](https://replicate.com/jyoung105/playground-v2.5/badge)](https://replicate.com/jyoung105/playground-v2.5/)

## Reference

[![Project](https://img.shields.io/badge/Project-8A2BE2)](https://playground.com/blog/playground-v2-5)
[![arXiv](https://img.shields.io/badge/arXiv-2402.17245v1-b31b1b.svg)](https://arxiv.org/html/2402.17245v1)
[![Hugging Face](https://img.shields.io/badge/🤗%20Huggingface-Model-yellow)](playgroundai/playground-v2.5-1024px-aesthetic)

## Example

1. A bustling Moroccan marketplace at sunset, with vibrant stalls displaying colorful textiles, spices, and lanterns, as merchants and shoppers engage in lively exchanges.
![Alt](../../../assets/pg251.png)

2. An underwater scene featuring a sunken pirate ship surrounded by coral reefs, schools of tropical fish, and a curious sea turtle exploring the wreckage.
![Alt](../../../assets/pg252.png)

3. A steampunk-inspired airship soaring above a Victorian-era city, with intricate gears and steam engines visible, and a crew of adventurers on deck.
![Alt](../../../assets/pg253.png)

4. A tranquil Scandinavian village during winter, with snow-covered rooftops, smoke rising from chimneys, and the Northern Lights illuminating the night sky.
![Alt](../../../assets/pg254.png)

5. A mystical desert landscape with towering sand dunes, an ancient, weathered statue half-buried in the sand, and a lone traveler approaching on camelback under a star-filled sky.
![Alt](../../../assets/pg255.png)

## Abstract

(Summarized by GPT-4o)

Playground has announced the release of **Playground v2.5**, an advanced text-to-image generative model that offers significant enhancements in aesthetic quality. 

**Key Features:**

* **Enhanced Color and Contrast:** Playground v2.5 addresses previous limitations by delivering images with more vibrant colors and improved contrast, resulting in more visually compelling outputs. 

* **Improved Multi-Aspect Ratio Generation:** The model effectively generates high-quality images across various aspect ratios, accommodating diverse user needs and applications. 

* **Refined Human-Centric Fine Details:** Playground v2.5 focuses on producing more realistic and detailed human features, enhancing the overall quality of human-centric images. 

The development of Playground v2.5 involved maintaining the existing SDXL architecture while implementing novel methods to achieve significant improvements in aesthetic quality. User studies indicate that Playground v2.5 outperforms leading open-source models like SDXL, Playground v2, and PixArt-α, as well as closed-source models such as DALL·E 3 and Midjourney v5.2. 

Playground v2.5 is available for public use on [Playground.com](https://playground.com/), and the aligned weights have been open-sourced on [HuggingFace](https://huggingface.co/). The release includes a license that facilitates use by research teams, reflecting Playground's commitment to contributing to the research and open-source communities. 

For a comprehensive understanding of the research process, techniques, and user evaluation methods employed in developing Playground v2.5, interested readers are encouraged to consult the detailed [technical report](https://marketing-cdn.playground.com/research/pgv2.5_compressed.pdf).  

## TODO
- [x] Inference code
- [ ] Method overview
- [ ] Train code
- [ ] Accelerate inference
- [x] Reduce memory usage
- [ ] Train LoRA, ControlNet, IPAdapter

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
cog run scripts/download-weights
```

4. save pipeline before deployment
```
cog run scripts/save-weights
```

5. predict to inference
```
cog predict -i prompt="an illustration of a man with hoodie on"
```