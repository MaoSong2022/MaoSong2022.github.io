---
title: Qwen-LLM技术报告总结
description: Qwen技术报告总结
date: 2025-07-03 10:47:27+0800
tags: 
    - Qwen
categories:
    - LLM 
---

# Introduction

Qwen在23年9月份发布了Qwen系列大语言模型，包括1.8B， 7B，14B三个size，作者还基于Qwen，构建了Code-Qwen-Chat，Math-Qwen-Chat等系列领域大语言模型。
作者在技术报告里介绍了架构，数据，评估。

# Pre-training

## Data

数据一共使用了3T token，主要是public web documents, encyclopedia, books, codes, etc，覆盖了中文和英文两种语言

数据处理：

1. 语言识别
2. 去重，包括MinHash和LSH算法
3. 质量过滤，包括基于规则和和基于ML的方法
4. 上采样，特定数据会进行上采样
5. 加入指令数据，提高模型的zero-shot和few-shot表现

## Tokenization

BPE tokenizer，最终的tokenizer大小为152K

## Architecture

模型架构基于[[LLaMA]]， 改动：

1. tie embdding: input embdding和output embdding使用的权重相同
2. position encoding:RoPE, inverse frequency的精度为FP32
3. bias: 取消了大部分的bias，增加了QKV bias，来提高模型的外推能力
4. Pre-Norm & RMSNorm
5. Activation function: SwiGLU

## Training

- 上下文长度：2048
- attention：flash attention
- optimizer：[[AdamW]]， $\beta_1=0.9$, $\beta_2=0.95$, $\epsilon=10^{-8}$.
- dtype: BF16

## Context extention

使用了三个技巧：

1. NTK-aware position interpolation
2. log-N scaling
3. window attention

observation: lower layer对上下文长度扩展更敏感, 因此作者动态调整了window size

# Post-training

包括SFT和RLHF两个阶段

### SFT

data： 使用了ChatML格式

### RLHF

[[PPO]]算法

reward model构建：基于Qwen-base model

RL训练：先更新value model 50 steps

发现：top-p设置为0.9比设置为1.0更好

### Tool-use and agent

作者使用了self-instruct来进行SFT，基于ReAct构建数据，数据包括2000条高质量数据

# Specialization

### code-qwen

code-qwen基于qwen continue Pretraining得到，continue Pretraining使用了900B的token。
然后基于code-qwen进行sft得到code-qwen-chat，包括7B和14B两个size

### math-qwen

基于qwen直接SFT得到，包括7B和14B两个size

# References

- [Length exploration](https://spaces.ac.cn/archives/9444)
