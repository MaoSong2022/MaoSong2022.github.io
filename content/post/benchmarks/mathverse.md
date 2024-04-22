---
title: MathVerse Does Your Multi-modal LLM Truly See the Diagrams in Visual Math Problems?
description: Paper reading notes on MathVerse
date: 2024-04-22 18:04:14+0800
tags:
    - math
categories:
    - Benchmark
    - Multimodal Large Language Model 
---

# TLDR
This paper proposes an multimodal large language model (MLLM) benchmark MathVerse for evaluating the mathmatical problem-solving capability of MLLMs. The author also proposes Chain of Thought (CoT) for fine-grained evaluation. Results show that MLLMs are not able to use the visual information, and even performs better wihtout visual input.


# Introduction
Multiple benchmarks have been proposed to evaluate the mathmathcal reasoning capability of MLLMs, such as GeoQA, UniGeo, MathVista and MMMU. 
However, the problem of existing benchmarks are:
1. The MLLMs may depend on text input instead of visual inputs to solve the problems.
2. The evaluation process is a black-box, we do not know at which step the MLLM makes a mistake.

The author proposes MathVerse to solve the aforementioned problems.


# Method
## Data collection
MathVerse contains $2612$ visual math problems and can be divided into plan geometry ($1746$), solid geometry ($332$) and functions ($534$).

## Text input processing
To check if MLLMs use visual input to solve the mathmatical problems, the author decompose the text input into three categories:
1. Descriptive information: Directly observable and clearly portrayed content in the diagram.
2. Implicit Property: Higher level of visual perception but less mathematical knowledge.
3. Essential Condition: specific numerical or algebraic measurements.

The dataset is then augmented by creating six versions of each problem:
- Text-dominant version: all text input and visual inputs are kept.
- Text-lite version: description information is discarded.
- Text-only version: visual input is discarded
- Vision-intensive version: description information and implicit property are discarded.
- Vision-dominant version:  description information and essential condition are discarded.
- Vision-only version: text input is discarded

## CoT evaluation
To visualize the reasoning process of MLLM when solving mathematcal problems, the author employ GPT-4 to generate key steps of solving the problem. Then scores are given to each reasoning step, in this way, the reasoning process can be tested. 


# Result
Experiment results show that:
1. MLLMs rely more on description information than seeing diagrams
2. LLMs Achieve Competitive Results to MLLMs.


# Reference
- [Arxiv](https://arxiv.org/pdf/2403.14624.pdf)
- [Gthub](https://mathverse-cuhk.github.io/)