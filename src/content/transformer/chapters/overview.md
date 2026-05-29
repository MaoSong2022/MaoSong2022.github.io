
## Introduction

### Why Transformer

transformer 的突破性改进在于架构范式的根本性转变。

虽然 attention 机制在之前的 RNN 模型中已经存在，但是 attention 主要作为一个辅助组建用于解决长距离遗忘问题，模型的主体依然是串行计算的 RNN

transformer 的改进体现在三点：

1. self-attention: 它让序列中任意两个位置的计算路径缩短为 1，从根本上解决了长距离依赖问题
2. parallelism: transformer 彻底抛弃了 RNN 的时序依赖，使得模型可以进行高效分布式训练
3. 架构的纯粹性：transformer 说明通过简单叠加 self-attention 与 FFN，配合残差链接与位置编码，构建了一个结构性更强，扩展性更高的统一架构
