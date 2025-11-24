# `nn.functional.causal_softmax`

对最后一维应用因果掩码 Softmax，用于自回归注意力场景。函数定义位于 `InfiniCore/python/infinicore/nn/functional.py`。

## 函数签名

```python
def causal_softmax(input: Tensor, out: Optional[Tensor] = None) -> Tensor
```

- `input`：任意维度张量，末维视为序列维，将应用因果掩码。
- `out`：可选输出张量，若提供需与 `input` 形状、`dtype`、`device` 完全一致。

默认返回新张量；提供 `out` 时调用 `_infinicore.causal_softmax_` 原位写入。

## 示例

```python
import infinicore as ic
from infinicore.nn import functional as F

device = ic.device("cuda:0")
logits = ic.empty((4, 128), dtype=ic.float16, device=device)

probs = F.causal_softmax(logits)
F.causal_softmax(logits, out=logits)  # 原位写回
```

## 相关链接

- [`nn.functional` 文档](../README.md)
- [`infinicore.attention` 算子](../../../ops/attention/README.md)
