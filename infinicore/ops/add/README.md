# `infinicore::op::add`

逐元素加法算子，支持广播与非连续张量。实现位于 `InfiniCore/src/infinicore/ops/add/add.cc`，头文件定义于 `InfiniCore/include/infinicore/ops/add.hpp`。

## 函数签名

```cpp
namespace infinicore::op {
    Tensor add(Tensor a, Tensor b);
    void add_(Tensor c, Tensor a, Tensor b);
    Tensor operator+(Tensor a, Tensor b);
}
```

- `a`：左操作数张量。
- `b`：右操作数张量，可与 `a` 形状相同或可广播到 `a`。
- `c`：输出张量（in-place 模式），需与结果形状、`dtype`、`device` 一致。

`add` 函数创建并返回新张量；`add_` 函数在原地写入结果；`operator+` 是 `add` 的运算符重载。

## 示例

```cpp
#include <infinicore.hpp>

using namespace infinicore;

Device device(Device::Type::NVIDIA, 0);
context::setDevice(device);

Tensor a = Tensor::ones({4, 8}, DataType::F16, device);
Tensor b = Tensor::ones({1, 8}, DataType::F16, device);  // 可广播

// Out-of-place
Tensor out = op::add(a, b);

// In-place
op::add_(a, b);  // a = a + b

// 运算符重载
Tensor c = a + b;
```

## 相关链接

- [`infinicore::op` 索引](../README.md)
- [`Tensor` 构造函数](../../tensor/README.md#构造函数)
