# `infinicore.nn.RMSNorm`

RMS 归一化层模块，对最后一个维度进行归一化。定义于 `InfiniCore/python/infinicore/nn/modules/normalization.py`。

## 类定义

```python
class RMSNorm(Module):
    def __init__(
        self,
        normalized_shape: Union[int, list[int]],
        eps=1e-6,
        elementwise_affine=True,
        device=None,
        dtype=None,
    ) -> None
```

## 概述

RMSNorm（Root Mean Square Layer Normalization）是对最后一个维度进行归一化的层。与 LayerNorm 不同，RMSNorm 不减去均值，也不使用偏置。

公式：`y = (x / RMS(x)) * weight`

其中 `RMS(x) = sqrt(mean(x^2) + eps)`

RMSNorm 在 LLaMA、Galactica 等现代语言模型中用作 LayerNorm 的更简单、更快的替代方案。

## 构造函数参数

- `normalized_shape`：要归一化的特征维度大小（可以是整数或列表）。
- `eps`：数值稳定性常数，默认为 `1e-6`。
- `elementwise_affine`：是否使用逐元素仿射变换，必须为 `True`。
- `device`：权重所在的设备。
- `dtype`：权重的数据类型。

## 主要方法

- `forward(x)`：前向传播，对最后一个维度应用 RMSNorm。

## 属性

- `weight`：权重张量，形状为 `normalized_shape`。

## 输入输出形状

- 输入：`(*, normalized_shape)`，其中 `*` 是任意数量的维度。
- 输出：与输入形状相同的归一化张量。

归一化应用于最后一个维度。例如：
- 输入：`[batch, seq_len, hidden_size]` -> 对 `hidden_size` 维度归一化
- 输入：`[batch, hidden_size]` -> 对 `hidden_size` 维度归一化

## 示例

```python
import infinicore as ic
from infinicore.nn import RMSNorm

device = ic.device("cuda:0")

# 创建 RMSNorm，隐藏层大小 4096
norm = RMSNorm(4096, eps=1e-6, device=device, dtype=ic.float16)

# 输入：形状 [batch, seq_len, hidden_size] = [2, 10, 4096]
input = ic.empty((2, 10, 4096), dtype=ic.float16, device=device)

# 输出：形状 [batch, seq_len, hidden_size] = [2, 10, 4096]
output = norm(input)

# 访问权重
weight = norm.weight  # shape: (4096,)
```

## 相关链接

- [`nn.modules` 模块概览](../README.md)
- [`nn.functional.rms_norm`](../../functional/rms_norm/README.md)
- [`Module` 基类](../module/README.md)
