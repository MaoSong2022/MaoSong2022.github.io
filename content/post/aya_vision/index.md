---
title: Aya Vision模型总结
description: 包含8B, 32B两个size，支持23种语言
date: 2025-03-17 17:58:24+0800

tags: 
    - multilingual
categories:
    - MLLM 
---

# 介绍

Aya Vision是一个多模态大语言模型，包含8B, 32B两个size，支持23种语言。Aya Vision基于 Aya Expanse大语言模型。

# 模型架构

Aya Vision的模型架构如下图所示

![Aya Vision模型架构 [1]](architecture.png)

- Vision Encoder: SigLip2-patch14-384
- Vision-text connector: 2 layer MLP
- LLM: Aya Expanse 8B/ 32B

# 训练

训练包含两个stage

# References

1. [Aya Vision Blog](https://huggingface.co/blog/aya-vision)
