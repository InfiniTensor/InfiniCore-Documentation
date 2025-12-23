# `infinicore::op::matmul`

矩阵乘法/GEMM 算子，实现位于 `InfiniCore/src/infinicore/ops/matmul/matmul.cc`，头文件定义于 `InfiniCore/include/infinicore/ops/matmul.hpp`。

## 函数签名

```cpp
namespace infinicore::op {
    Tensor matmul(Tensor a, Tensor b, float alpha = 1.0f);
    void matmul_(Tensor c, Tensor a, Tensor b, float alpha = 1.0f);
}
```

- `a`：左乘矩阵，形状需满足 GEMM 要求，可包含批维。
- `b`：右乘矩阵，与 `a` 的维度兼容。
- `c`：输出张量（in-place 模式），需与结果形状、`dtype`、`device` 完全一致。
- `alpha`：缩放因子，默认为 1.0。

`matmul` 函数创建并返回新张量；`matmul_` 函数在原地写入结果。底层会复用 InfiniOP GEMM 描述符完成计算。

## 输入要求

- 支持常见数据类型（如 `F16`、`F32`、`BF16`）。
- 支持批量维度；所有批维需两输入对齐。
- 当 `c` 与输入不处于同一设备或数据类型不匹配时，底层会抛出异常。

## 示例

```cpp
#include <infinicore.hpp>

using namespace infinicore;

Device device(Device::Type::NVIDIA, 0);
context::setDevice(device);

Tensor a = Tensor::ones({4, 8}, DataType::F16, device);
Tensor b = Tensor::ones({8, 16}, DataType::F16, device);

// Out-of-place
Tensor c = op::matmul(a, b);

// In-place
op::matmul_(c, a, b);

// 带缩放因子
Tensor d = op::matmul(a, b, 2.0f);
```

## 相关链接

- [`infinicore::op` 索引](../README.md)
- [`Tensor` 构造函数](../../tensor/README.md#构造函数)
