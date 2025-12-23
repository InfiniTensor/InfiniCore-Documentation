# `infinicore::op` 顶层算子

该模块通过 C++ 接口将常用算子直接暴露在 `infinicore::op` 命名空间下，实现位于 `InfiniCore/src/infinicore/ops/`，头文件定义于 `InfiniCore/include/infinicore/ops/`。所有函数均支持 in-place 和 out-of-place 两种模式。

## 通用注意事项

- 所有参数必须是 `infinicore::Tensor` 实例。
- 所有输入张量必须位于同一设备上。
- in-place 操作（函数名以 `_` 结尾）将结果写入第一个参数。
- out-of-place 操作会创建并返回新张量。
- 算子执行依赖底层运行时；请在创建张量时传入期望的 `device`，保持所有输入处于同一设备上。

## 文档索引

- [`add`](add/README.md)
- [`matmul`](matmul/README.md)
- [`rearrange`](rearrange/README.md)
- [`attention`](attention/README.md)

## 示例

```cpp
#include <infinicore.hpp>

using namespace infinicore;

Device device(Device::Type::NVIDIA, 0);
context::setDevice(device);

Tensor a = Tensor::ones({4, 8}, DataType::F16, device);
Tensor b = Tensor::ones({4, 8}, DataType::F16, device);

// Out-of-place 操作
Tensor c = op::add(a, b);

// In-place 操作
op::add_(a, b);  // a = a + b

// 矩阵乘法
Tensor d = op::matmul(a, b->permute({1, 0}));  // (4, 8) @ (8, 4)

// 重排为连续存储
Tensor contiguous = op::rearrange(d);

// Attention
Tensor q = Tensor::empty({8, 1, 128}, DataType::F16, device);
Tensor k = Tensor::empty({2, 1, 128}, DataType::F16, device);
Tensor v = Tensor::empty({2, 1, 128}, DataType::F16, device);
Tensor k_cache = Tensor::empty({2, 128, 128}, DataType::F16, device);
Tensor v_cache = Tensor::empty({2, 128, 128}, DataType::F16, device);
Tensor attn_out = op::attention(q, k, v, k_cache, v_cache, 0);
```

## 相关链接

- [`C++ API 总览`](../README.md)
- [`Tensor` 文档](../tensor/README.md)
