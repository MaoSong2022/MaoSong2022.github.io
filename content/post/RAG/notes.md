---
title: Notes on RAG
description: Design for AI agentic workflows
date: 2024-04-14 12:38:04+0800
image: agent_performance.png

tags: 
    - RAG
categories:
    - RAG
    - Large Language Model 
---


## Problems of LLM
- Out of date knowledge: the model cannot gain knowledge after training
- Humiliation: the model may generate nonsense output
- Specific domain: the generalized model is difficult to adapt to specific domain
- Enthetic problems: the model may encounter 

## Fine-tuning
Fine-tuning is used to improve performance of foundation model on specific tasks with the help with some supervised data

Fine-tuning methods can be classified into:
1. Based on range of updated parameters:
    - Full Model fine-tuning: update the parameters of the whole model
    - Partial fine-tuning: freeze the top layer; freeze the bottom layer
2. Based on special technology:
    - Adapter tuning
    - LoRA
    - Continual Learning fine-tuning
3. Based on input:
    - Instruction tuning
4. Based on objective
    - Multi-task fine-tuning

Problems of fine-tuning:
1. Requires task-specific labeled data, may cause overfitting and catastrophic forgetting。
2. The generalization ability is limited, and fine-tuning are required when adapting to new tasks
3. The performance may be destroyed after fine-tuning, for example, safety.


## RAG
problems of RAG
1. The quality of retrieval
    - The retrieved text cannot be aligned with the queried text.
    - The queried text are not retrieved all.
    - Redundancy or out-dated data may cause inaccuracy.
2. the quality of response generation
    - Model Humiliation
    - Irrelevance
    - Organize the output to make it reasonable
    - Depends on the external information


## Advance RAG


## Module RAG


## Other technologies
1. Query transformations
2. Sentence window retrieval
3. Fusion retrieval/ hybrid search
4. multi-document agents