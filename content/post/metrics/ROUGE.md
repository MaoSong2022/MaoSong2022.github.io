---
title: ROUGE (Recall-Oriented Understudy)
date: 2024-05-09 17:35:20+0800
description: The metric that evaluates similarity between summaries.
tags: 
    - Metric
categories:
    - Large Language Model 
    - Natural Language Processing
math: true
---

ROUGE stands for Recall-Oriented Understudy for Gisting Evaluation. It includes several automatic evaluation methods that measure the similarity between summaries.

# Preliminaries

# ROUGE-N: N-gram co-occurrence statistics

## Single reference

ROUGE-N is an n-gram recall between a candidate summary $\hat{y}$ and a set of reference summaries $S=\{y_1,\dots,y_n\}$. ROUGE-N is defined as follows:

$$ \text{ROUGE-N}(\hat{y}, S) = \frac{\sum_{i=1}^n\sum_{s\in G_n(y_i)}\max_{y\in S_i}C(\hat{y},s)}{\sum_{i=1}^n\sum_{s\in G_n(y_i)}C(\hat{y},s)} $$

Features of ROUGE-N:

1. the denominator increases as we add more references, since there might exists multiple good summaries.
2. A candidate summary that contains words shared by more references is favored by the ROUGE-N measure.

# ROUGE-L

# ROUGE-W

# ROUGE-S

# Reference

- [ROUGE: A Package for Automatic Evaluation of Summaries](https://aclanthology.org/W04-1013.pdf)
