# Playground-v2.0

Playground v2.0: A diffusion-based text-to-image generation model trained from scratch by the research team at Playground

[![Replicate](https://replicate.com/jyoung105/playground-v2.0/badge)](https://replicate.com/jyoung105/playground-v2.0/)

## Reference

[![Project](https://img.shields.io/badge/Project-8A2BE2)](https://playground.com/blog/playground-v2)
[![Hugging Face](https://img.shields.io/badge/🤗%20Huggingface-Model-yellow)](https://huggingface.co/playgroundai/playground-v2-1024px-aesthetic)

## Example

1. A bustling Moroccan marketplace at sunset, with vibrant stalls displaying colorful textiles, spices, and lanterns, as merchants and shoppers engage in lively exchanges.
![Alt](../../../assets/pg21.png)

2. An underwater scene featuring a sunken pirate ship surrounded by coral reefs, schools of tropical fish, and a curious sea turtle exploring the wreckage.
![Alt](../../../assets/pg22.png)

3. A steampunk-inspired airship soaring above a Victorian-era city, with intricate gears and steam engines visible, and a crew of adventurers on deck.
![Alt](../../../assets/pg23.png)

4. A tranquil Scandinavian village during winter, with snow-covered rooftops, smoke rising from chimneys, and the Northern Lights illuminating the night sky.
![Alt](../../../assets/pg24.png)

5. A mystical desert landscape with towering sand dunes, an ancient, weathered statue half-buried in the sand, and a lone traveler approaching on camelback under a star-filled sky.
![Alt](../../../assets/pg25.png)

## Abstract

(Summarized by GPT-4o)

The article titled "Playground v2: A new leap in creativity" announces the release of Playground v2, an advanced graphics model developed by Playground. 

**Key Highlights:**

* **Open-Source Availability:** Playground v2 is available for public use on [playground.com](https://playground.com/) and can be downloaded from [HuggingFace](https://huggingface.co/). 

* **Commercial Use Permitted:** Users are allowed to utilize Playground v2 for commercial purposes, providing flexibility for various applications. 

* **Benchmark Performance:** Early benchmarks indicate that Playground v2 is preferred 2.5 times more than Stable Diffusion XL across thousands of prompts, highlighting its superior performance. 

* **MJHQ-30K Benchmark:** The introduction of the MJHQ-30K benchmark allows for automatic evaluation of a model’s aesthetic quality by computing the Fréchet Inception Distance (FID) on a high-quality dataset curated from Midjourney. 

* **Pre-Trained Weights Release:** To support research in environments with limited computational resources, Playground has released pre-trained weights of the model in 256px and 512px stages on HuggingFace. 

The article also acknowledges the contributions of the Playground Research Team and invites individuals interested in advancing computer graphics to explore career opportunities with Playground.  

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
cd ./cog-diffusers/Normal/Playground/v20
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