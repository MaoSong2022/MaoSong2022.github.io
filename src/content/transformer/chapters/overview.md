## Introduction

Transformer 将 attention 从循环网络中的辅助模块变为序列建模的核心，通过堆叠 self-attention 与前馈网络构建模型。本文围绕 Qwen3 风格的 decoder-only Transformer，自顶向下介绍各模块的职责和核心代码。

### Why Transformer

早期循环网络按时间步依次更新隐藏状态；attention 可以帮助模型直接访问序列中的相关位置。Transformer 进一步采用以 attention 为主体的架构，主要有以下特点：

1. **直接建立 token 之间的联系**：在可见范围内，self-attention 让一个位置直接聚合其他位置的信息，缩短远距离位置之间的信息传递路径。这有助于建模长距离依赖，但不意味着长上下文问题已经完全解决。
2. **适合并行训练**：给定完整输入序列时，同一层中不同位置的表示可以并行计算。自回归生成仍需要逐 token 进行，因此训练时的并行性与生成过程需要区分。
3. **统一的模块化结构**：Attention、FFN、normalization、残差连接和位置编码各有明确职责，便于逐层堆叠，也便于替换模块、调整模型规模。

### 阅读路线

本文先介绍整体架构，再沿着前向传播的数据流展开：

- **CausalLM**：用语言模型输出头将隐藏状态映射到词表 logits。
- **Transformer 与 DecoderLayer**：组织 embedding、多层计算、normalization 和残差连接。
- **RMSNorm、MLP 与 Attention**：理解每个基础模块的计算及代码。
- **Position Encoding**：介绍位置编码，并对照两种 RoPE 实现。
- **模型类型**：比较 encoder-only、decoder-only 和 encoder-decoder 的组织方式。

代码以教学为目的保留核心结构，省略部分初始化、接口和工程细节。理解架构后，可以继续阅读 [LLM Resource Computation](/blog/llm-resource-computation/)，推导这些模块的参数量、FLOPs 以及训练和推理显存。
