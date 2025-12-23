# `infinicore` C++ 前端

*InfiniCore* 提供了高性能的 C++ 前端，实现位于 `InfiniCore/src/infinicore/`，头文件定义于 `InfiniCore/include/infinicore/`。该模块提供了核心张量、算子、设备上下文等功能的 C++ 接口，便于在推理框架或高性能应用中直接集成。

## 模块结构

| 符号 | 说明 |
| --- | --- |
| `Device` | 设备句柄类，实现位于 `src/infinicore/device.cc`，头文件 `include/infinicore/device.hpp`，支持 CPU、CUDA、MLU、NPU、MUSA 等多种设备类型。 |
| `DataType` / `dtype` | 数据类型枚举，实现位于 `src/infinicore/dtype.cc`，头文件 `include/infinicore/dtype.hpp`，包括各种整数、浮点数和复数类型。 |
| `Tensor` / `TensorImpl` | 张量类，实现位于 `src/infinicore/tensor/`，头文件 `include/infinicore/tensor.hpp`，提供张量的创建、操作和视图功能。 |
| `Memory` | 内存管理类，实现位于 `src/infinicore/memory.cc`，头文件 `include/infinicore/memory.hpp`，封装设备内存的分配和释放。 |
| `DeviceEvent` | 设备事件类，实现位于 `src/infinicore/device_event.cc`，头文件 `include/infinicore/device_event.hpp`，用于同步和性能测量。 |
| `context` | 运行时上下文命名空间，实现位于 `src/infinicore/context/`，头文件 `include/infinicore/context/context.hpp`，提供设备管理、内存分配、流同步等功能。 |
| 顶层算子（`add`、`matmul`、`rearrange`、`attention` 等） | 暴露在 `infinicore::op` 命名空间下，实现位于 `src/infinicore/ops/`，头文件 `include/infinicore/ops/`。 |
| `infinicore::nn` | 神经网络相关模块集合，实现位于 `src/infinicore/nn/`，头文件 `include/infinicore/nn/`，包括线性层、嵌入层、RMSNorm 等。 |

所有符号定义在 `infinicore` 命名空间下，可通过 `#include <infinicore.hpp>` 统一引入。

## API 索引

- [`Device`](device/README.md)
- [`DataType`](dtype/README.md)
- [`Tensor`](tensor/README.md)
- [`Memory`](memory/README.md)
- [`DeviceEvent`](device_event/README.md)
- [`context`](context/README.md)
- [`ops`](ops/README.md)
  - [`add`](ops/add/README.md)
  - [`matmul`](ops/matmul/README.md)
  - [`rearrange`](ops/rearrange/README.md)
  - [`attention`](ops/attention/README.md)
- [`nn`](nn/README.md)
  - [`nn::Linear`](nn/linear/README.md)
  - [`nn::Embedding`](nn/embedding/README.md)
  - [`nn::RMSNorm`](nn/rmsnorm/README.md)

## 张量与构造函数

`Tensor` 是对底层 `TensorImpl` 的智能指针包装，常用接口包括：

- `shape()` / `ndim()` / `size(dim)` / `stride(dim)`：获取张量维度与步长信息。
- `dtype()` / `device()`：返回数据类型与设备。
- `numel()` / `is_contiguous()`：查看张量元素数量与存储布局。
- `copy_from(src)` / `to(device)`：执行数据拷贝与跨设备搬运。
- `contiguous()` / `permute(dims)` / `view(shape)` / `as_strided(size, stride)`：布局调整与视图操作。
- `debug(filename)`：将张量内容打印或输出到文件。

常用构造函数包括 `Tensor::empty`、`Tensor::strided_empty`、`Tensor::zeros`、`Tensor::ones`、`Tensor::from_blob`、`Tensor::strided_from_blob` 等：

```cpp
#include <infinicore.hpp>

using namespace infinicore;

Device cpu = Device::cpu();
Tensor a = Tensor::empty({4, 8}, DataType::F16, cpu);
Tensor b = Tensor::ones({4, 8}, DataType::F16, cpu);
a->copy_from(b);
```

> 注意：这些函数要求显式传入 `DataType` 与 `Device`，避免隐式推断。

## 顶层算子 (`infinicore::*`)

详见 [`ops` 文档索引](ops/README.md) 及各算子文档。

## 神经网络模块 (`infinicore::nn`)

详见 [`nn` 文档](nn/README.md) 及各模块文档。

## 运行时上下文

- `context` 命名空间在进程内维护运行时状态；创建张量时请显式传入 `device`，并保持算子的所有输入位于同一设备。
- 如需强制同步，可调用 `context::syncStream()`、`context::syncDevice()` 等函数。
- 在同一执行流内串行调用算子通常无需额外同步。

## 端到端示例

```cpp
#include <infinicore.hpp>

using namespace infinicore;

Device device(Device::Type::NVIDIA, 0);
context::setDevice(device);

Tensor q = Tensor::empty({8, 1, 128}, DataType::F16, device);
Tensor k = Tensor::empty({2, 1, 128}, DataType::F16, device);
Tensor v = Tensor::empty({2, 1, 128}, DataType::F16, device);
Tensor k_cache = Tensor::empty({2, 128, 128}, DataType::F16, device);
Tensor v_cache = Tensor::empty({2, 128, 128}, DataType::F16, device);

Tensor out = attention(q, k, v, k_cache, v_cache, 0);

out->debug();
```

## 相关链接

- [`infinicore.ops 顶层算子`](ops/README.md)
- [`nn 神经网络模块`](nn/README.md)
- [`InfiniOP` 统一算子库](/infiniop/README.md)
