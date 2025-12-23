# `infinicore.nn.Embedding`

嵌入层模块，将索引映射到密集向量。定义于 `InfiniCore/python/infinicore/nn/modules/sparse.py`。

## 类定义

```python
class Embedding(Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx=None,
        max_norm=None,
        norm_type=2.0,
        scale_grad_by_freq=False,
        sparse=False,
        _weight=None,
        _freeze=False,
        device=None,
        dtype=None,
    ) -> None
```

## 概述

`Embedding` 是一个简单的查找表，存储固定字典和大小的嵌入向量。该模块通常用于存储词嵌入，并通过索引检索它们。

## 构造函数参数

- `num_embeddings`：嵌入字典的大小（词汇表大小）。
- `embedding_dim`：每个嵌入向量的维度。
- `padding_idx`：填充索引（当前不支持）。
- `max_norm`：最大范数（当前不支持）。
- `norm_type`：范数类型（当前不支持）。
- `scale_grad_by_freq`：是否按频率缩放梯度（当前不支持）。
- `sparse`：是否使用稀疏梯度（当前不支持）。
- `device`：嵌入权重所在的设备。
- `dtype`：嵌入权重的数据类型。

**注意**：当前版本仅支持基本参数，其他参数（`padding_idx`、`max_norm` 等）必须为默认值。

## 主要方法

- `forward(input)`：前向传播，根据索引查找对应的嵌入向量。

## 属性

- `weight`：嵌入权重张量，形状为 `(num_embeddings, embedding_dim)`。

## 输入输出形状

- 输入：索引张量，可以是任意形状 `(*)`，通常是 `[batch_size]` 或 `[batch_size, seq_len]`。
- 输出：嵌入向量张量，形状为 `(*, embedding_dim)`，其中 `*` 与输入形状匹配。

## 示例

```python
import infinicore as ic
from infinicore.nn import Embedding

device = ic.device("cuda:0")

# 创建嵌入层：10000 个词，300 维嵌入
embedding = Embedding(10000, 300, device=device, dtype=ic.float16)

# 输入：形状 [batch_size, seq_len] = [2, 5]
indices = ic.empty((2, 5), dtype=ic.int64, device=device)
# ... 填充索引值 ...

# 输出：形状 [batch_size, seq_len, embedding_dim] = [2, 5, 300]
embeddings = embedding(indices)

# 访问权重
weight = embedding.weight  # shape: (10000, 300)
```

## 相关链接

- [`nn.modules` 模块概览](../README.md)
- [`nn.functional.embedding`](../../functional/embedding/README.md)
- [`Module` 基类](../module/README.md)
