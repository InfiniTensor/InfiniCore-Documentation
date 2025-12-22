# `infinicore::Memory`

内存管理类，实现位于 `InfiniCore/src/infinicore/memory.cc`，头文件定义于 `InfiniCore/include/infinicore/memory.hpp`，用于封装设备内存的分配和释放。

## 概述

`Memory` 类封装了设备内存的底层管理，包括内存指针、大小、设备信息和删除器。它通常不直接使用，而是通过 `Tensor` 间接管理。

## 构造方式

```cpp
#include <infinicore/memory.hpp>

using namespace infinicore;

// 通常通过 context 命名空间分配
std::shared_ptr<Memory> mem = context::allocateMemory(size);
```

## 主要方法

- `data()`：返回指向内存数据的 `std::byte*` 指针。
- `device()`：返回内存所在的设备。
- `size()`：返回内存大小（字节数）。
- `is_pinned()`：判断是否为固定内存（pinned memory）。

## 内存分配

通常通过 `context` 命名空间的函数分配内存：

- `context::allocateMemory(size)`：在当前设备上分配内存。
- `context::allocateHostMemory(size)`：分配主机内存。
- `context::allocatePinnedHostMemory(size)`：分配固定主机内存。

## 示例

```cpp
#include <infinicore.hpp>

using namespace infinicore;

Device device(Device::Type::NVIDIA, 0);
context::setDevice(device);

// 通常通过 Tensor 间接使用 Memory
Tensor tensor = Tensor::empty({4, 8}, DataType::F16, device);
// tensor 内部使用 Memory 管理内存
```

## 相关链接

- [`Tensor` 文档](../tensor/README.md)
- [`context` 运行时上下文](../context/README.md)
