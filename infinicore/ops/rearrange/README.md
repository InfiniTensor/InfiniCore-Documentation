# `infinicore::op::rearrange`

重排算子，将非连续张量转换为连续存储。实现位于 `InfiniCore/src/infinicore/ops/rearrange/rearrange.cc`，头文件定义于 `InfiniCore/include/infinicore/ops/rearrange.hpp`。

## 函数签名

```cpp
namespace infinicore::op {
    Tensor rearrange(Tensor x);
    void rearrange_(Tensor y, Tensor x);
}
```

- `x`：输入张量，可以是连续或非连续的。
- `y`：输出张量（in-place 模式），需与 `x` 的形状、`dtype`、`device` 一致。

`rearrange` 函数创建并返回连续存储的新张量；`rearrange_` 函数将结果写入 `y`。如果输入已经是连续的，可能会直接返回或复制输入。

## 示例

```cpp
#include <infinicore.hpp>

using namespace infinicore;

Device device(Device::Type::NVIDIA, 0);
context::setDevice(device);

Tensor a = Tensor::empty({4, 8}, DataType::F16, device);
Tensor b = a->permute({1, 0});  // 非连续

// Out-of-place
Tensor contiguous = op::rearrange(b);

// In-place
Tensor c = Tensor::empty(b->shape(), b->dtype(), b->device());
op::rearrange_(c, b);
```

## 相关链接

- [`infinicore::op` 索引](../README.md)
- [`Tensor` 视图操作](../../tensor/README.md#视图操作)
