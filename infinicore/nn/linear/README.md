# `infinicore::nn::Linear`

线性层，实现 `output = input @ weight.T + bias`。实现位于 `InfiniCore/src/infinicore/nn/linear.cc`，头文件定义于 `InfiniCore/include/infinicore/nn/linear.hpp`。

## 类定义

```cpp
namespace infinicore::nn {
    class Linear : public BaseLinear {
    public:
        Linear(size_t in_features, size_t out_features, bool bias = true,
               const DataType &dtype = DataType::F32, const Device &device = Device());
        
        Tensor forward(Tensor &input) const;
        std::string extra_repr() const;
    };
    
    class ColumnParallelLinear : public BaseLinear { /* ... */ };
    class RowParallelLinear : public BaseLinear { /* ... */ };
}
```

## 构造函数参数

- `in_features`：输入特征数。
- `out_features`：输出特征数。
- `bias`：是否使用偏置，默认为 `true`。
- `dtype`：权重和偏置的数据类型，默认为 `DataType::F32`。
- `device`：权重和偏置所在的设备。

## 主要方法

- `forward(input)`：前向传播，计算 `input @ weight.T + bias`。
- `forward(input, residual)`：带残差连接的前向传播（InfiniLM 风格），计算 `input @ weight.T + bias + residual`。
- `weight()`：获取权重张量。
- `bias()`：获取偏置张量。
- `in_features()`：获取输入特征数。
- `out_features()`：获取输出特征数。
- `has_bias()`：检查是否有偏置。

## 并行版本

- `ColumnParallelLinear`：列并行线性层，支持张量并行。
- `RowParallelLinear`：行并行线性层，支持张量并行。

## 示例

```cpp
#include <infinicore.hpp>

using namespace infinicore;
using namespace infinicore::nn;

Device device(Device::Type::NVIDIA, 0);
context::setDevice(device);

// 创建线性层
Linear linear(128, 64, true, DataType::F16, device);

// 前向传播
Tensor input = Tensor::empty({32, 128}, DataType::F16, device);
Tensor output = linear.forward(input);

// 带残差连接
Tensor residual = Tensor::empty({32, 64}, DataType::F16, device);
Tensor output_with_residual = linear.forward(input, residual);

// 访问参数
Tensor weight = linear.weight();
Tensor bias = linear.bias();
```

## 相关链接

- [`nn` 模块概览](../README.md)
- [`Module` 基类](../module/README.md)
