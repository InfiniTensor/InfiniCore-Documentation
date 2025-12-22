# `infinicore::Tensor` 模块

`Tensor` 类及核心构造函数，实现位于 `InfiniCore/src/infinicore/tensor/`，头文件定义于 `InfiniCore/include/infinicore/tensor.hpp`，负责在 C++ 端封装底层 `TensorImpl` 并提供常用操作。

## `Tensor` 类

`Tensor` 是对 `TensorImpl` 的智能指针包装，通过 `operator->()` 访问底层实现。

### 主要属性

- `shape()`（`const Shape &`）/ `ndim()` / `size(dim)`：获取张量维度信息。
- `stride(dim)`：返回指定维度的步长。
- `strides()`：返回所有维度的步长。
- `dtype()`：对应的标量类型（参见 [`DataType` 文档](../dtype/README.md)）。
- `device()`：张量所在设备（参见 [`Device` 文档](../device/README.md)）。
- `numel()`：元素总数。
- `is_contiguous()`：判断是否连续存储。
- `element_size()`：单个元素占用的字节数。
- `nbytes()`：张量数据占用的总字节数。
- `is_pinned()`：判断是否为固定内存（pinned memory）。
- `desc()`：返回底层 `infiniopTensorDescriptor_t` 描述符。

### 常用方法

- `copy_from(src)`：将 `src` 的数据拷贝到当前张量。
- `to(device)`：执行跨设备转换，返回新 `Tensor`。
- `as_strided(size, stride)`：创建共享存储的视图。
- `contiguous()`：返回连续副本。
- `permute(dims)`：重新排列维度。
- `view(shape)`：改变张量形状（需满足可视条件）。
- `unsqueeze(dim)`：在指定维度插入大小为 1 的维度。
- `narrow(slices)`：在指定维度上缩小张量。
- `debug(filename)`：打印张量信息或将原始数据写入文件。
- `info()`：返回张量的字符串描述。

## 构造函数

全部为 `Tensor` 类的静态方法，默认要求显式传入 `DataType` 与 `Device`：

```cpp
#include <infinicore.hpp>

using namespace infinicore;
```

### 说明

- `Tensor::empty(shape, dtype, device, pin_memory=false)`：按给定形状分配未初始化存储。
- `Tensor::strided_empty(shape, strides, dtype, device, pin_memory=false)`：按指定步长分配存储。
- `Tensor::zeros(shape, dtype, device, pin_memory=false)`：创建全零张量。
- `Tensor::ones(shape, dtype, device, pin_memory=false)`：创建全一张量。
- `Tensor::from_blob(data_ptr, shape, dtype, device)`：将外部内存包装为 `Tensor`，不接管内存所有权。
- `Tensor::strided_from_blob(data_ptr, shape, strides, dtype, device)`：同上但可显式指定步长。

## 视图操作

视图操作返回共享底层存储的新张量：

- `view(new_shape)`：改变形状，要求元素总数一致。
- `permute(order)`：重新排列维度顺序。
- `unsqueeze(dim)`：在指定位置插入大小为 1 的维度。
- `narrow(slices)`：在指定维度上缩小范围。
- `as_strided(new_shape, new_strides)`：创建具有指定形状和步长的视图。

## 数据转移

- `to(device)`：将张量转移到指定设备，返回新张量。
- `copy_from(src)`：从源张量拷贝数据到当前张量（形状必须匹配）。

## 示例

```cpp
#include <infinicore.hpp>

using namespace infinicore;

Device device(Device::Type::NVIDIA, 0);
context::setDevice(device);

Tensor a = Tensor::empty({4, 8}, DataType::F16, device);
Tensor b = Tensor::ones({4, 8}, DataType::F16, device);

a->copy_from(b);
Tensor c = a->permute({1, 0})->contiguous();

// 视图操作
Tensor d = c->view({8, 4});
Tensor e = d->unsqueeze(0);  // shape: [1, 8, 4]

// 调试输出
a->debug();  // 打印到标准输出
a->debug("tensor.bin");  // 保存到文件
```

## 相关链接

- [`infinicore` 模块概览](../README.md)
- [`顶层算子`](../ops/README.md)
- [`nn` 模块](../nn/README.md)
