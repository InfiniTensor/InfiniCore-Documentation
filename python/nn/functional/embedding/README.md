# `infinicore.nn.functional.embedding`

嵌入查找函数式接口，从权重矩阵中查找嵌入向量。定义于 `InfiniCore/python/infinicore/nn/functional/embedding.py`。

## 函数签名

```python
def embedding(
    input: Tensor,
    weight: Tensor,
    padding_idx=None,
    max_norm=None,
    norm_type=2.0,
    scale_grad_by_freq=False,
    sparse=False,
    *,
    out=None,
) -> Tensor
```

## 参数说明

- `input`：索引张量，可以是任意形状 `(*)`，通常是 `[batch_size]` 或 `[batch_size, seq_len]`。**必须位于 CPU 设备上**。
- `weight`：嵌入权重矩阵，形状为 `(num_embeddings, embedding_dim)`。
- `padding_idx`：填充索引（当前不支持，必须为 `None`）。
- `max_norm`：最大范数（当前不支持，必须为 `None`）。
- `norm_type`：范数类型（当前不支持）。
- `scale_grad_by_freq`：是否按频率缩放梯度（当前不支持，必须为 `False`）。
- `sparse`：是否使用稀疏梯度（当前不支持，必须为 `False`）。
- `out`：可选输出张量，若提供需与结果形状、`dtype`、`device` 一致。

## 返回值

返回嵌入向量张量，形状为 `(*, embedding_dim)`，其中 `*` 与输入形状匹配。

## 注意事项

- **输入索引张量必须位于 CPU 设备上**。
- 当前版本仅支持基本参数，其他参数（`padding_idx`、`max_norm` 等）必须为默认值。

## 示例

```python
import infinicore as ic
from infinicore.nn import functional as F

device = ic.device("cuda:0")
cpu_device = ic.device("cpu")

# 创建嵌入权重（在 GPU 上）
weight = ic.empty((10000, 300), dtype=ic.float16, device=device)

# 输入索引（必须在 CPU 上）
indices = ic.empty((2, 5), dtype=ic.int64, device=cpu_device)
# ... 填充索引值 ...

# 查找嵌入向量
embeddings = F.embedding(indices, weight)  # shape: (2, 5, 300)
```

## 相关链接

- [`nn.functional` 函数式接口](../README.md)
- [`nn.Embedding` 模块](../../modules/sparse/README.md)
