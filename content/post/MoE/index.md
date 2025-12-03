---
title: MoE tutorial
description: 关于LLM中MoE架构的一个tutorial
date: 2025-10-23 16:13:29+0800
lastmod: 2025-10-23 16:13:29+0800
math: true
tags: 
    - tutorial
    - MoE
categories:
    - LLM
---



## 介绍

MoE 模型是一个将 transformer block 中 FFN 替换为 MoE layer 的方法，通过 MoE，我们可以让模型在相同的激活参数下，达到更好的性能。

本文中，我们基于主流的 MoE 模型学习一下 MoE 的方法与进展，更多细节请参阅参考文献。

## 方法

MoE 模型和 dense 模型的示意图如下，图源 [olmoe](https://maosong.website/p/notes-on-olmoe/)

![[MoE_architecture.png]]

一个 MoE layer 包括两个模块：

1. Router：Router 负责为 token 指定合适的专家
2. Expert：Expert 负责处理 token

对于输入 $x\in\mathbb{R}^d$, 我们假设有 $N$ 个 Expert，router 一般是一个 linear layer 再加上一个 softmax，其构建了 $\mathbb{R}^d\to\mathbb{R}^N$ 的映射，其定义为：

$$
G(x) = \mathrm{softmax}(W_gx + b)\in\mathbb{R}^N
$$

其中 $W_g\in\mathbb{R}^{N\times d}$, $b\in\mathbb{R}^N$ 是可学习的参数。$G(x)\in\mathbb{R}^N$ 是一个概率分布，$G_{i}$ 代表了第 $i$ 个 Expert 对于当前 token $x$ 的重要性.

 一般来说，Expert 会使用和 dense 模型一样的 MLP，即使用 SwiGLU 激活函数的 FFN，见 [[Assignment 1]] ， 我们记为

$$
E_i(x) = FFN(x), i = 1,\dots,N
$$

接下来，基于 $G(x)$ 和 $E_i(x)$, 我们会使用合适的方法来挑选 $K<N$ 个 Expert 出来，其中 $K>0$ 是给定的超参数，我们记挑选出来的 $K$ 个 Expert 的 index 为 $e_1,\dots,e_K$, 则我们最终 MoE layer 的输出为

$$
y = \sum_{i=1}^K\mathrm{Normalize}(G_{e_i}) E_{e_i}(x)
$$

这里 $\mathrm{Normalize}(\cdot)$ 代表我们对于输出进行归一化，即

$$
\mathrm{Normalize}(G_{e_i}) = \frac{\exp(G_{e_i})}{\sum_{i=1}^K \exp(G_{e_i})}
$$

## 代码

我们这里展示基于 [olmoe](https://maosong.website/p/notes-on-olmoe/) 的代码，代码如下

```python
class OlmoeSparseMoeBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.gate = nn.Linear(config.hidden_size, self.num_experts, bias=False)
        self.experts = nn.ModuleList([OlmoeMLP(config) for _ in range(self.num_experts)])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        # hidden_states: (batch * sequence_length, hidden_dim)
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        # routing_weights: (batch * sequence_length, top_k)
        # selected_experts: indices of top_k experts
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be selected
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits
```

【TODO】理解后面代码优化的部分

## Variant

在构建 MoE Layer 的过程中，有以下设计方法。

### Routing Type

在为专家分配 token 的时候，我们有如下方式：

1. 为每个 token 选取若干个专家
2. 为每个专家选取若个个 token
3. 动态分配 token 与专家之间的关系

三种选择方式如下图所示，图源 [MoE survey](https://arxiv.org/pdf/2209.01667)

![[MoE_routing.png]]

图源：【参考文献 2】

1. Token Choice: 每个 token 选取 top-k 的专家，好处是每个 token 都会被处理，缺点是负载不均衡
2. Expert Choice: 每个专家选取 top-k 的 token，此时每个专家处理的 token 个数是相同的，这个方法的好处是 load balance。缺点是自回归生成的方式没有完整序列长度的信息，从而导致 token dropping，也就是某些 token 不会被任何专家处理，某些 token 会被多个专家处理
3. Global Choice: 全局分配决定 token 和专家的匹配关系

现在几乎所有的模型都选择方式 1，即每个 token 选取 top-k 的专家。 [olmoe](https://maosong.website/p/notes-on-olmoe/) 对比了以下方式 1 和方式 2 的表现，如下图所示

![MoE routing strategy EC v.s. TC](olmoe-routing-strategy.png)

可以看到，加入 load balancing loss 之后，相比于 Expert Choice, Token Choice 的表现更好。但是，expert choice 更加高效，作者认为 expert choice 更适用于多模态，因为丢掉 noise image tokens 对 text token 影响会比较小。因此，在 olmoe 中，作者使用 token choice 作为 routing 策略

### Granularity of Expert

[[DeepSeekMoE]]

[[olmoe]]

### Shared Expert

Shared Expert 由 [[DeepSeekMoE]] 提出，其基本思想为，固定某几个专家，响应所有的 token，这样可以让某些专家学习到共有的知识，而让其他的专家学习到特定的知识。这个方法随后被 [[Qwen1.5]], [[Qwen2]] , [[Qwen2.5]] 以及 [[DeepSeek-V3]] 所采用。

[[DeepSeekMoE]] 给出的实验结果如下

![DeepSeek-Moe shared experts ablation study](DeepSeekMoE-ablation-experts.png)

作者发现，当使用 shared experts 之后，模型在大部分 benchmark 上的表现都有所提升。

[[olmoe]] 在 32 个专家下进行了实验，比较了 4 个激活专家和 3 个激活专家 +1 个共享专家两种设置的表现，结果如下：

![Olmoe shared experts performance](olmoe-shared-experts.png)

作者认为，加入 shared experts 之后，组合的可能性有所减少，这会降低模型的泛化性。因此，在 olmoe 中，作者没有使用 shared experts.

虽然 [[Qwen1.5]], [[Qwen2]] 和 [[Qwen2.5]] 都使用了 shared experts, 但是后续的 [[Qwen3]] 中却并没有使用，作者并未解释原因。

## Training

训练的时候，我们必须保证 sparsity，但是 sparsity 意味着不可微，为了解决不可微的问题，现有解决方法：

1. 基于 RL 的算法
2. 随机扰动
3. balancing loss

### Backpropogation

我们假设损失函数为 $\mathcal{L}=g(y)$, 则

$$
\frac{\partial \mathcal{L}}{\partial W_g} = \frac{\partial \mathcal{L}}{\partial g}\left(\sum_{i=1}^K E_{e_i}(x)\frac{\partial G_{e_i}}{\partial W_g}+\sum_{i=1}^K G_{e_i}(x)\frac{\partial E_{e_i}}{\partial W_g}\right)
$$

其中，第二部分关于专家部分的反向传播是可微的，我们直接忽略。在第一项中，我们发现， $\frac{\partial G_{e_i}}{\partial W_g}$ 是不可微的, 因此我们需要解决这个不可微的问题。

解决方案一般有以下几种

#### REINFORCE

#### Straight Through Estimator

#### Noisy Top-k Gating

#### Differentiable Top-k Relaxations

Gumbel-Softmax (or Concrete Distribution)

### Load Balancing Loss

见 [Load Balancing loss](Load%20Balancing%20loss.md)

[[olmoe]]

### Router Z-loss

Router z-loss 用于提升 MoE 模型训练的稳定性和最终表现，z-loss 会惩罚 gating 网络中较大的 logits，因为这些较大的 logits 会导致数值溢出，给定一个 batch $B$, 对于 router layer 输入的 logits $x_i$, 其定义如下：

$$
\mathcal{L}_{z}(x) = \frac{1}{B}\sum_{i=1}^B\left(\log \sum_{j=1}^K \exp(x_j^{(i)})\right)^2
$$

即再求和之前，先计算对应的数值，然后给较大的数值一个更大的惩罚，这样可以让每个 token 对专家的 logits 分布更加平滑，而不是仅关注少数几个专家

实验结果【olmoe 图 11】

[[olmoe]]

### Upcycling

upsampling 是一个将 dense model 转化为 MoEmodel 的方法，具体做法就是我们复制 dense model 中的 FFN layer 得到对应 MoE layer 中的 Expert，然后我们再结合 router 训练，这样可以提高整体的训练效率。

但是这样做的问题是，MoE 模型会被 dense 模型的一些超参数所限制

实验结果【olmoe 图 8】

[[MiniCPM]]

[[Qwen1.5]]

[[Mixtral MoE]]

## Pros and Cons

优点

- MoE 在训练和推理效率等方面具有优势
- 相同的激活参数下，MoE 模型表现的更好

缺点：

- 训练不稳定
- 在相同存储量下的模型性能以及下游任务小样本微调的表现上存在劣势
- 更高的内存占用

Dense 模型：

- 相同总参数量下稠密模型的性能更强，对于探索模型能力上限的意义更为重大

## MoE 模型

[[LLaMA4]]

[[Mistral-7B]]

[[DeepSeekMoE]]

[[DeepSeek-V3]]

[[olmoe]]

[[Grok系列]]

## 结论

在本文中，我们系统性回顾了 MoE 的相关概念，MoE 模型已经是现在大语言模型的主流架构，比如商业模型 [[Gemini2.5]], 开源领先的模型 [[DeepSeek-V3]] , [[LLaMA4]] 以及 [[Qwen3]] 等都采用了 MoE 的架构，如何进一步优化 MoE 的训练方式是当前研究的一个重点方向。

## References

- [Switch Transformer](https://arxiv.org/pdf/2101.03961)
- [MoE Survey](https://arxiv.org/pdf/2209.01667)
- [olmoe](https://openreview.net/forum?id=xXTkbTBmqq)
- [GShard](https://arxiv.org/pdf/2006.16668)
- [blog](https://www.cnblogs.com/rossiXYZ/p/18800825)
- [MoE a big data perspective](https://arxiv.org/pdf/2501.16352v1)
- [Mixture of Experts Explained](https://huggingface.co/blog/moe)
