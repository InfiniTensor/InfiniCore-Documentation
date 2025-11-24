# `nn.functional.silu`

Sigmoid Linear Unit (SiLU) 激活函数。实现位于 `InfiniCore/python/infinicore/nn/functional.py`。

## 函数签名

```python
def silu(
    input: Tensor,
    inplace: bool = False,
    *,
    out: Optional[Tensor] = None,
) -> Tensor
```

- `input`：待激活张量。
- `inplace`：是否原地写回到 `input`。
- `out`：可选输出张量，若提供需与 `input` 形状、`dtype`、`device` 一致。

### 行为说明

- 当 `inplace=True` 时，直接调用 `_infinicore.silu_` 写回 `input` 并返回。
- 当未提供 `out` 且 `infinicore.use_ntops=True` 且设备类型为 `"cuda"` 或 `"musa"` 时，会委托 `ntops.torch.silu` 以复用优化实现。
- 其他情况下调用 `_infinicore.silu` 或 `_infinicore.silu_` 完成计算。

## 示例

```python
import infinicore as ic
from infinicore.nn import functional as F

device = ic.device("cuda:0")
x = ic.ones((4, 8), dtype=ic.float16, device=device)

y = F.silu(x)                 # 返回新张量
F.silu(x, inplace=True)       # 原地更新
```

## 相关链接

- [`nn.functional` 文档](../README.md)
- [`use_ntops` 协作说明](../../../README.md#与-ntops-的协作)
