# `infinicore::nn` 模块

`infinicore::nn` 聚合了面向神经网络的辅助模块，实现位于 `InfiniCore/src/infinicore/nn/`，头文件定义于 `InfiniCore/include/infinicore/nn/`，包括模块基类、参数管理、线性层、嵌入层等组件。

## 模块结构

| 子模块 | 说明 |
| --- | --- |
| `Module` | 模块基类，提供参数管理、状态字典、子模块注册等功能。 |
| `Parameter` | 参数类，用于标识可训练参数，继承自 `Tensor`。 |
| `Linear` | 线性层，实现 `output = input @ weight.T + bias`。 |
| `ColumnParallelLinear` | 列并行线性层，支持张量并行。 |
| `RowParallelLinear` | 行并行线性层，支持张量并行。 |
| `Embedding` | 嵌入层，将索引映射到密集向量。 |
| `RMSNorm` | RMS 归一化层，对最后一个维度进行归一化。 |

## 使用示例

### 模块化构建

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
```

### 自定义模块

```cpp
#include <infinicore.hpp>

using namespace infinicore;
using namespace infinicore::nn;

class MyModel : public Module {
protected:
    INFINICORE_NN_MODULE(Linear, layer1);
    INFINICORE_NN_MODULE(Linear, layer2);
    INFINICORE_NN_PARAMETER(scaling_factor);

public:
    MyModel() {
        INFINICORE_NN_MODULE_INIT(layer1, 128, 64);
        INFINICORE_NN_MODULE_INIT(layer2, 64, 32);
        INFINICORE_NN_PARAMETER_INIT(scaling_factor, ({1}, DataType::F32, Device()));
    }

    Tensor forward(Tensor &input) {
        Tensor x = layer1_->forward(input);
        x = layer2_->forward(x);
        // 使用 scaling_factor
        return x;
    }
};
```

## 相关链接

- [`Module` 基类](module/README.md)
- [`Parameter` 参数类](parameter/README.md)
- [`Linear` 线性层](linear/README.md)
- [`Embedding` 嵌入层](embedding/README.md)
- [`RMSNorm` RMS 归一化](rmsnorm/README.md)
- [`C++ API 总览`](../README.md)
