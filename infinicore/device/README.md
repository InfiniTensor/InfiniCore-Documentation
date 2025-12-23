# `infinicore::Device`

设备句柄类，实现位于 `InfiniCore/src/infinicore/device.cc`，头文件定义于 `InfiniCore/include/infinicore/device.hpp`，用于在 C++ 端选择和描述运行时设备。

## 构造方式

```cpp
#include <infinicore.hpp>

using namespace infinicore;

Device cpu;                              // 默认 CPU
Device cuda0(Device::Type::NVIDIA, 0);  // NVIDIA GPU 0
Device cuda1(Device::Type::NVIDIA, 1);  // NVIDIA GPU 1
Device cpu_device = Device::cpu();       // 静态工厂方法
```

- `Type`：支持 `CPU`、`NVIDIA`、`CAMBRICON`、`ASCEND`、`METAX`、`MOORE`、`ILUVATAR`、`KUNLUN`、`HYGON`、`QY` 等，具体取决于编译时支持。
- `Index`：设备序号，默认为 0。
- `Device::cpu()`：静态工厂方法，返回 CPU 设备。

## 主要方法

- `getType()`：获取设备类型。
- `getIndex()`：获取设备序号。
- `toString()`：返回字符串表示，如 `"NVIDIA:0"`。
- `toString(const Type &type)`：静态方法，将设备类型转换为字符串。
- `operator==()` / `operator!=()`：设备比较操作。

## 与运行时的关系

- `Device` 实例用于所有张量构造函数及顶层算子，确保在正确的设备上执行。
- 使用 `context::setDevice(device)` 设置当前设备。

## 示例

```cpp
#include <infinicore.hpp>

using namespace infinicore;

Device device(Device::Type::NVIDIA, 0);
context::setDevice(device);

Tensor tensor = Tensor::empty({4, 8}, DataType::F16, device);
```

## 相关链接

- [`Tensor` 构造函数](../tensor/README.md#构造函数)
- [`运行时上下文`](../context/README.md)
