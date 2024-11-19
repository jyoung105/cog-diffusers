# Kolors

Kolors: Effective Training of Diffusion Model for Photorealistic Text-to-Image Synthesis

[![Replicate](https://replicate.com/jyoung105/kolors/badge)](https://replicate.com/jyoung105/kolors/)

## Reference

[![Project](https://img.shields.io/badge/Project-8A2BE2)](https://kwai-kolors.github.io/)
[![arXiv](https://img.shields.io/badge/arXiv-pdf-b31b1b.svg)](https://github.com/Kwai-Kolors/Kolors/blob/master/imgs/Kolors_paper.pdf)
[![GitHub](https://img.shields.io/badge/GitHub-Repo-181717?logo=github)](https://github.com/Kwai-Kolors/Kolors)
[![Hugging Face](https://img.shields.io/badge/🤗%20Huggingface-Model_1-yellow)](https://huggingface.co/Kwai-Kolors/Kolors)
[![Hugging Face - diffusers](https://img.shields.io/badge/🤗%20Huggingface-Model_2-yellow)](https://huggingface.co/Kwai-Kolors/Kolors-diffusers)

## Example

1. A bustling Moroccan marketplace at sunset, with vibrant stalls displaying colorful textiles, spices, and lanterns, as merchants and shoppers engage in lively exchanges.
![Alt](../../assets/kolors1.png)

2. An underwater scene featuring a sunken pirate ship surrounded by coral reefs, schools of tropical fish, and a curious sea turtle exploring the wreckage.
![Alt](../../assets/kolors2.png)

3. A steampunk-inspired airship soaring above a Victorian-era city, with intricate gears and steam engines visible, and a crew of adventurers on deck.
![Alt](../../assets/kolors3.png)

4. A tranquil Scandinavian village during winter, with snow-covered rooftops, smoke rising from chimneys, and the Northern Lights illuminating the night sky.
![Alt](../../assets/kolors4.png)

5. A mystical desert landscape with towering sand dunes, an ancient, weathered statue half-buried in the sand, and a lone traveler approaching on camelback under a star-filled sky.
![Alt](../../assets/kolors5.png)

## Explanation

(Summarized by GPT-4o)

The paper titled "Kolors: Effective Training of Diffusion Model for Photorealistic Text-to-Image Synthesis" introduces **Kolors**, a large-scale text-to-image generation model developed by the Kuaishou Kolors team. Kolors is trained on billions of text-image pairs and supports both Chinese and English inputs, demonstrating strong performance in understanding and generating content in both languages. 

**Key Features:**

* **Bilingual Capability:** Kolors is designed to handle both Chinese and English inputs, making it versatile for a wide range of applications. 

* **High Visual Quality:** The model exhibits significant advantages over both open-source and closed-source models in visual quality, complex semantic accuracy, and text rendering for both Chinese and English characters. 

* **Comprehensive Evaluation:** Kolors has been evaluated using a dataset named KolorsPrompts, which includes over 1,000 prompts across 14 categories and 12 evaluation dimensions. The evaluation process incorporates both human and machine assessments, demonstrating Kolors' highly competitive performance. 

* **Open-Source Commitment:** The Kolors team has open-sourced various components, including inference checkpoints, LoRA, ControlNet (Pose, Canny, Depth), IP-Adapter, ComfyUI, and Diffusers, facilitating further research and application in the field. 

The paper provides detailed insights into the training process, evaluation metrics, and potential applications of Kolors, contributing to the advancement of photorealistic text-to-image synthesis.  

## TODO
- [x] Inference code
- [ ] Method overview
- [ ] Train code
- [ ] Accelerate inference
- [ ] Reduce memory usage
- [ ] Train LoRA, ControlNet, IPAdapter

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