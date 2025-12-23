# `infinicore.nn.RoPE`

旋转位置嵌入（Rotary Position Embedding）模块。定义于 `InfiniCore/python/infinicore/nn/modules/rope.py`。

## 类定义

```python
class RoPE(Module):
    def __init__(
        self,
        max_position_embeddings: int,
        rope_theta: float,
        head_dim: int,
        device=None,
        dtype=None,
    )
```

## 概述

RoPE（Rotary Position Embedding）是一种位置编码方法，通过旋转矩阵将位置信息编码到注意力机制中。广泛应用于 GPT-J、LLaMA 等现代语言模型。

## 构造函数参数

- `max_position_embeddings`：最大序列长度。
- `rope_theta`：RoPE 的基础周期（base period）。
- `head_dim`：注意力头的维度。
- `device`：sin/cos 表所在的设备。
- `dtype`：sin/cos 表的数据类型。

## 主要方法

- `forward(states, position_ids, algo=RopeAlgo.GPT_NEOX)`：前向传播，应用旋转位置嵌入。

## 输入输出形状

- 输入 `states`：`(bs, seq_len, num_heads, head_dim)`。
- 输入 `position_ids`：`(bs, seq_len)`。
- 输出：`(bs, seq_len, num_heads, head_dim)`，与输入 `states` 形状相同。

## 算法类型

- `RopeAlgo.GPT_J`：GPT-J 风格的 RoPE 算法。
- `RopeAlgo.GPT_NEOX`：GPT-NeoX 风格的 RoPE 算法（默认）。

## 示例

```python
import infinicore as ic
from infinicore.nn import RoPE
from infinicore.nn.functional import RopeAlgo

device = ic.device("cuda:0")

# 创建 RoPE 模块
rope = RoPE(
    max_position_embeddings=2048,
    rope_theta=10000.0,
    head_dim=128,
    device=device,
    dtype=ic.float16
)

# 输入状态和位置 ID
states = ic.empty((2, 10, 8, 128), dtype=ic.float16, device=device)
position_ids = ic.empty((2, 10), dtype=ic.int64, device=device)
# ... 填充 position_ids ...

# 应用 RoPE
output = rope(states, position_ids, algo=RopeAlgo.GPT_NEOX)
```

## 相关链接

- [`nn.modules` 模块概览](../README.md)
- [`nn.functional.rope`](../../functional/rope/README.md)
- [`Module` 基类](../module/README.md)
