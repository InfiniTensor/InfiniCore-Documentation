# `nn.functional.swiglu`

SwiGLU（Swish-Gated Linear Unit）函数式实现，定义在 `InfiniCore/python/infinicore/nn/functional.py`。

## 函数签名

```python
def swiglu(
    input: Tensor,
    other: Tensor,
    *,
    out: Optional[Tensor] = None,
) -> Tensor
```

- `input`：激活分支张量。
- `other`：门控分支张量。要求与 `input` 在形状、`dtype`、`device` 完全一致。
- `out`：可选输出张量，若提供需满足上述一致性条件。

内部调用 `_infinicore.swiglu` / `_infinicore.swiglu_` 完成计算；未提供 `out` 时返回新张量，否则原地写入。

## 示例

```python
import infinicore as ic
from infinicore.nn import functional as F

device = ic.device("cuda:0")
a = ic.ones((4, 8), dtype=ic.float16, device=device)
b = ic.ones((4, 8), dtype=ic.float16, device=device)

out = F.swiglu(a, b)
F.swiglu(a, b, out=a)  # 原位更新
```

## 相关链接

- [`nn.functional` 文档](../README.md)
- [`infinicore.ops` 索引](../../../ops/README.md)
