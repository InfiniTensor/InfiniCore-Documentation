# `infinicore::nn::RMSNorm`

RMS 归一化层，对最后一个维度进行归一化。实现位于 `InfiniCore/src/infinicore/nn/rmsnorm.cc`，头文件定义于 `InfiniCore/include/infinicore/nn/rmsnorm.hpp`。

## 类定义

```cpp
namespace infinicore::nn {
    class RMSNorm : public Module {
    public:
        RMSNorm(size_t normalized_shape,
                double eps = 1e-6,
                const DataType &dtype = DataType::F32,
                const Device &device = Device());
        
        Tensor forward(const Tensor &x) const;
        std::string extra_repr() const;
        
        Tensor weight() const;
        size_t normalized_shape() const;
        double eps() const;
    };
}
```

## 概述

RMSNorm（Root Mean Square Layer Normalization）是对最后一个维度进行归一化的层。与 LayerNorm 不同，RMSNorm 不减去均值，也不使用偏置。

公式：`y = (x / RMS(x)) * weight`

其中 `RMS(x) = sqrt(mean(x^2) + eps)`

RMSNorm 在 LLaMA、Galactica 等现代语言模型中用作 LayerNorm 的更简单、更快的替代方案。

## 构造函数参数

- `normalized_shape`：要归一化的特征维度大小（通常是隐藏层大小）。
- `eps`：数值稳定性常数，默认为 `1e-6`。
- `dtype`：权重的数据类型，默认为 `DataType::F32`。
- `device`：权重所在的设备。

## 主要方法

- `forward(x)`：前向传播，对最后一个维度应用 RMSNorm。
- `weight()`：获取权重张量。
- `normalized_shape()`：获取归一化的特征维度大小。
- `eps()`：获取 epsilon 值。

## 输入输出

- 输入：形状为 `(*, normalized_shape)` 的张量，其中 `*` 是任意数量的维度。
- 输出：与输入形状相同的归一化张量。

归一化应用于最后一个维度。例如：
- 输入：`[batch, seq_len, hidden_size]` -> 对 `hidden_size` 维度归一化
- 输入：`[batch, hidden_size]` -> 对 `hidden_size` 维度归一化

## 示例

```cpp
#include <infinicore.hpp>

using namespace infinicore;
using namespace infinicore::nn;

Device device(Device::Type::NVIDIA, 0);
context::setDevice(device);

// 创建 RMSNorm，隐藏层大小 4096
RMSNorm norm(4096, 1e-6, DataType::F16, device);

// 输入：形状 [batch, seq_len, hidden_size] = [2, 10, 4096]
Tensor input = Tensor::empty({2, 10, 4096}, DataType::F16, device);
// ... 填充输入值 ...

// 输出：形状 [batch, seq_len, hidden_size] = [2, 10, 4096]
Tensor output = norm.forward(input);

// 访问权重
Tensor weight = norm.weight();
```

## 相关链接

- [`nn` 模块概览](../README.md)
- [`Module` 基类](../module/README.md)
