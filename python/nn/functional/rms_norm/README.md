# `nn.functional.rms_norm`

实现 Root Mean Square LayerNorm。函数定义位于 `InfiniCore/python/infinicore/nn/functional.py`。

## 函数签名

```python
def rms_norm(
    input: Tensor,
    normalized_shape: list[int],
    weight: Tensor,
    eps: float = 1e-5,
    *,
    out: Optional[Tensor] = None,
) -> Tensor
```

- `input`：待归一化张量，末维通常为隐藏维度。
- `normalized_shape`：期望归一化的维度大小列表，会与 `weight.shape` 进行严格比较。
- `weight`：缩放系数张量，维度需与 `normalized_shape` 匹配。
- `eps`：数值稳定项，默认 `1e-5`。
- `out`：可选输出张量；若提供需与 `input` 形状、`dtype`、`device` 一致。

函数首先断言 `normalized_shape == weight.shape`，然后调用 `_infinicore.rms_norm` / `_infinicore.rms_norm_` 完成计算。

## 示例

```python
import infinicore as ic
from infinicore.nn import functional as F

device = ic.device("cuda:0")
x = ic.ones((4, 1024), dtype=ic.float16, device=device)
gamma = ic.ones((1024,), dtype=ic.float16, device=device)

y = F.rms_norm(x, normalized_shape=list(gamma.shape), weight=gamma, eps=1e-5)
F.rms_norm(x, normalized_shape=[1024], weight=gamma, out=x)  # 原位写回
```

## 相关链接

- [`nn.functional` 文档](../README.md)
- [`Tensor` 构造函数](../../../README.md#张量与构造函数)
