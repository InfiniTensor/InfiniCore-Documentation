# `infinicore.nn.functional.linear`

线性变换函数式接口，实现 `y = x @ weight.T + bias`。定义于 `InfiniCore/python/infinicore/nn/functional/linear.py`。

## 函数签名

```python
def linear(
    input: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    *,
    out: Optional[Tensor] = None
) -> Tensor
```

## 参数说明

- `input`：输入张量，形状为 `(*, in_features)`，其中 `*` 表示任意数量的维度。
- `weight`：权重张量，形状为 `(out_features, in_features)`。
- `bias`：可选偏置张量，形状为 `(out_features,)`。
- `out`：可选输出张量，若提供需与结果形状、`dtype`、`device` 一致。

## 返回值

返回线性变换后的张量，形状为 `(*, out_features)`，其中 `*` 与输入形状匹配。

## 示例

```python
import infinicore as ic
from infinicore.nn import functional as F

device = ic.device("cuda:0")

# 输入和权重
input = ic.empty((32, 128), dtype=ic.float16, device=device)
weight = ic.empty((64, 128), dtype=ic.float16, device=device)
bias = ic.empty((64,), dtype=ic.float16, device=device)

# 线性变换
output = F.linear(input, weight, bias)  # shape: (32, 64)

# 不带偏置
output_no_bias = F.linear(input, weight)  # shape: (32, 64)
```

## 相关链接

- [`nn.functional` 函数式接口](../README.md)
- [`nn.Linear` 模块](../../modules/linear/README.md)
