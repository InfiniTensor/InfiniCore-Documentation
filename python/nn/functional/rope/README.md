# `infinicore.nn.functional.rope`

旋转位置嵌入（Rotary Position Embedding）函数式接口。定义于 `InfiniCore/python/infinicore/nn/functional/rope.py`。

## 函数签名

```python
def rope(
    x: Tensor,
    pos_ids: Tensor,
    sin_table: Tensor,
    cos_table: Tensor,
    algo: RopeAlgo = RopeAlgo.GPT_NEOX,
    *,
    out=None,
) -> Tensor
```

## 参数说明

- `x`：输入张量，形状为 `(bs, seq_len, num_heads, head_dim)`。
- `pos_ids`：位置 ID 张量，形状为 `(bs, seq_len)`。
- `sin_table`：正弦表，形状为 `(max_position, head_dim // 2)`。
- `cos_table`：余弦表，形状为 `(max_position, head_dim // 2)`。
- `algo`：RoPE 算法类型，默认为 `RopeAlgo.GPT_NEOX`。
- `out`：可选输出张量，若提供需与输入形状、`dtype`、`device` 一致。

## 算法类型

- `RopeAlgo.GPT_J`：GPT-J 风格的 RoPE 算法。
- `RopeAlgo.GPT_NEOX`：GPT-NeoX 风格的 RoPE 算法（默认）。

## 输入要求

- `x` 需要在维度 0 和维度 1 上连续（`seq_len * stride[1] == stride[0]`）。

## 示例

```python
import infinicore as ic
from infinicore.nn import functional as F
from infinicore.nn.functional import RopeAlgo

device = ic.device("cuda:0")

# 创建 sin/cos 表
max_position = 2048
head_dim = 128
sin_table = ic.empty((max_position, head_dim // 2), dtype=ic.float16, device=device)
cos_table = ic.empty((max_position, head_dim // 2), dtype=ic.float16, device=device)
# ... 填充 sin_table 和 cos_table ...

# 输入状态和位置 ID
x = ic.empty((2, 10, 8, 128), dtype=ic.float16, device=device)
pos_ids = ic.empty((2, 10), dtype=ic.int64, device=device)
# ... 填充 pos_ids ...

# 应用 RoPE
output = F.rope(x, pos_ids, sin_table, cos_table, algo=RopeAlgo.GPT_NEOX)
```

## 相关链接

- [`nn.functional` 函数式接口](../README.md)
- [`nn.RoPE` 模块](../../modules/rope/README.md)
