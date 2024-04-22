---
title: Introduction to Llama3
description: An brief introduction to Llama3
date: 2024-04-22 16:22:19+0800
tags: 
    - Llama
categories:
    - Large Language Model 
---

Meta released Llama3 at April 18, which is evaluated on several benchmarks and achieves the SOTA on open-sourced LLMs


# Model Architecture
Several improvements are made on Llama3 compared to llama2:
1. Llama3 uses a tokenizer with a vocabulary of 128K tokens.
2. Llama3 adopts grouped query attention (GQA) across both the 8B and 70B sizes.
3. Llama3 uses to context window of size 8192 tokens


# Traning
Llama3 uses 15T tokens for pre-training. Compares to Llama2, it is seven times larger and includes four times more code.

5% data of the training dataset are non-English to support multi-lingual use cases.

Data processing includes:
1. Heuristic filters
2. NSFW filters
3. Semantic deduplication approaches
4. Text classifiers to predict data quality. Llama2 is used to generate training data for the text classifiers.

Data mixing strategy is explored to improve the performance of Llama3.


# Scaling up pretraining
Llama3 developed a series of scaling laws for downstream benchmark evaluations.

Scaling laws help:
1. Select an optimal data mix and to make informed decisions on how to best use training compute.
2. Scaling laws allow Llama3 to predict the performance of the largest models on key tasks without training the models.

The authors finds our that the performance of the model continues to improve log-linearly as the training tokens increase. It is seen that  Larger models can match the performance of these smaller models with less training compute, but smaller models are generally preferred because they are much more efficient during inference.

The authors combine three types of parallelization:
1. Data parallelization
2. Model parallelization
3. Pipeline parallelization

# Instruction fine-tuning
The fine-tuning of Llama3 contains:
1. Supervised fine-tuning
2. Rejection sampling
3. Proximal Policy Optimization 
4. Direct Preference Optimization


Learning from perference rankings via PPO and DPO also greatly improved the performance of LLma3 on reasoning and coding tasks. Since perference ranking helps the model to select answer when it is in a dilemma.

