# `infinicore::nn::Parameter`

参数类，用于标识可训练参数，继承自 `Tensor`。实现位于 `InfiniCore/src/infinicore/nn/parameter.cc`，头文件定义于 `InfiniCore/include/infinicore/nn/parameter.hpp`。

## 类定义

```cpp
namespace infinicore::nn {
    class Parameter : public Tensor {
    public:
        Parameter();
        
        Parameter(const Tensor &tensor,
                  Size tp_dim = 0,
                  Size tp_rank = 0,
                  Size tp_size = 1);
        
        Parameter(const Shape &shape,
                  const DataType &dtype,
                  const Device &device,
                  Size tp_dim = 0,
                  Size tp_rank = 0,
                  Size tp_size = 1);
        
        void load_blob(const void *data);
        void load(const Tensor &tensor);
    };
}
```

## 概述

`Parameter` 继承自 `Tensor`，用于标识神经网络中的可训练参数。它支持张量并行配置，可以在分布式训练中使用。

## 构造函数

- `Parameter()`：创建空参数。
- `Parameter(tensor, tp_dim, tp_rank, tp_size)`：从张量创建参数，支持张量并行配置。
- `Parameter(shape, dtype, device, tp_dim, tp_rank, tp_size)`：创建指定形状的参数。

## 张量并行参数

- `tp_dim`：分片的维度。
- `tp_rank`：当前分片在张量并行组中的排名。
- `tp_size`：张量并行组的总分片数。

## 主要方法

- `load_blob(data)`：从内存块加载参数数据。
- `load(tensor)`：从张量加载参数数据。

## 示例

```cpp
#include <infinicore.hpp>

using namespace infinicore;
using namespace infinicore::nn;

Device device(Device::Type::NVIDIA, 0);
context::setDevice(device);

// 创建参数
Parameter weight({64, 128}, DataType::F16, device);

// 从内存块加载
float *data = new float[64 * 128];
// ... 填充数据 ...
weight.load_blob(data);

// 从张量加载
Tensor tensor = Tensor::empty({64, 128}, DataType::F16, device);
weight.load(tensor);

// 在模块中使用
class MyModule : public Module {
protected:
    INFINICORE_NN_PARAMETER(weight);
    
public:
    MyModule() {
        INFINICORE_NN_PARAMETER_INIT(weight, ({64, 128}, DataType::F16, Device()));
    }
};
```

## 相关链接

- [`nn` 模块概览](../README.md)
- [`Module` 基类](../module/README.md)
- [`Tensor` 文档](../../tensor/README.md)
