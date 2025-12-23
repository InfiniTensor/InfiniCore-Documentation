# `infinicore::DataType` 与标量类型

标量类型枚举和相关工具函数，实现位于 `InfiniCore/src/infinicore/dtype.cc`，头文件定义于 `InfiniCore/include/infinicore/dtype.hpp`。

## 类型列表

### 整数类型
- `DataType::BYTE`：字节类型
- `DataType::BOOL`：布尔类型
- `DataType::I8` / `DataType::I16` / `DataType::I32` / `DataType::I64`：有符号整数
- `DataType::U8` / `DataType::U16` / `DataType::U32` / `DataType::U64`：无符号整数

### 浮点类型
- `DataType::F8`：8 位浮点数
- `DataType::F16`：16 位浮点数（半精度）
- `DataType::F32`：32 位浮点数（单精度）
- `DataType::F64`：64 位浮点数（双精度）
- `DataType::BF16`：16 位 BFloat16

### 复数类型
- `DataType::C16`：16 位复数
- `DataType::C32`：32 位复数
- `DataType::C64`：64 位复数
- `DataType::C128`：128 位复数

## 工具函数

- `toString(const DataType &dtype)`：将数据类型转换为字符串表示。
- `dsize(const DataType &dtype)`：返回数据类型占用的字节数。

## 示例

```cpp
#include <infinicore.hpp>

using namespace infinicore;

Device cpu = Device::cpu();
Tensor a = Tensor::empty({4, 8}, DataType::F16, cpu);

if (a->dtype() == DataType::F16) {
    std::cout << "half precision tensor" << std::endl;
}

size_t element_size = dsize(DataType::F16);  // 返回 2
std::string dtype_str = toString(DataType::F16);  // 返回 "F16"
```

## 相关链接

- [`Tensor` 构造函数](../tensor/README.md#构造函数)
- [`infinicore` 模块概览](../README.md)
