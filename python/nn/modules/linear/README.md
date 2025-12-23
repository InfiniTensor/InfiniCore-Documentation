# `infinicore.nn.Linear`

线性层模块，实现 `output = input @ weight.T + bias`。定义于 `InfiniCore/python/infinicore/nn/modules/linear.py`。

## 类定义

```python
class Linear(Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        device=None,
        dtype=None,
    ) -> None
```

## 构造函数参数

- `in_features`：输入特征数。
- `out_features`：输出特征数。
- `bias`：是否使用偏置，默认为 `False`。
- `device`：权重和偏置所在的设备。
- `dtype`：权重和偏置的数据类型。

## 主要方法

- `forward(input)`：前向传播，计算 `input @ weight.T + bias`。

## 属性

- `weight`：权重张量，形状为 `(out_features, in_features)`。
- `bias`：偏置张量，形状为 `(out_features,)`（如果 `bias=True`）。

## 输入输出形状

- 输入：`(*, in_features)`，其中 `*` 表示任意数量的维度。
- 输出：`(*, out_features)`，除最后一个维度外与输入形状相同。

## 示例

```python
import infinicore as ic
from infinicore.nn import Linear

device = ic.device("cuda:0")

# 创建线性层
linear = Linear(128, 64, bias=True, device=device, dtype=ic.float16)

# 前向传播
input = ic.empty((32, 128), dtype=ic.float16, device=device)
output = linear(input)  # shape: (32, 64)

# 访问参数
weight = linear.weight  # shape: (64, 128)
bias = linear.bias      # shape: (64,)
```

## 相关链接

- [`nn.modules` 模块概览](../README.md)
- [`nn.functional.linear`](../../functional/linear/README.md)
- [`Module` 基类](../module/README.md)
