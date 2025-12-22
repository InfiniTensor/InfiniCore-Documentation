# `infinicore::DeviceEvent`

设备事件类，实现位于 `InfiniCore/src/infinicore/device_event.cc`，头文件定义于 `InfiniCore/include/infinicore/device_event.hpp`，用于同步操作和性能测量。

## 概述

`DeviceEvent` 类似于 CUDA 的 `cudaEvent_t`，提供以下功能：
- 在特定设备流上记录事件
- 与事件同步
- 测量事件之间的耗时
- 查询事件完成状态
- 让流等待事件

## 构造方式

```cpp
#include <infinicore.hpp>

using namespace infinicore;

Device device(Device::Type::NVIDIA, 0);
context::setDevice(device);

DeviceEvent event1;                              // 在当前设备上创建
DeviceEvent event2(device);                       // 在指定设备上创建
DeviceEvent event3(device, INFINIRT_EVENT_DISABLE_TIMING);  // 带标志创建
```

## 主要方法

- `record()`：在当前流上记录事件。
- `record(stream)`：在指定流上记录事件。
- `synchronize()`：等待事件完成（阻塞）。
- `query()`：检查事件是否已完成。
- `elapsed_time(other)`：计算与另一个事件之间的耗时（毫秒）。
- `wait(stream)`：让指定流等待此事件。
- `device()`：获取事件所在的设备。
- `get()`：获取底层事件句柄。
- `is_recorded()`：检查事件是否已被记录。

## 事件标志

- `INFINIRT_EVENT_DEFAULT`：默认标志
- `INFINIRT_EVENT_DISABLE_TIMING`：禁用计时
- `INFINIRT_EVENT_BLOCKING_SYNC`：使用阻塞同步

## 示例

```cpp
#include <infinicore.hpp>

using namespace infinicore;

Device device(Device::Type::NVIDIA, 0);
context::setDevice(device);

// 创建事件
DeviceEvent start_event(device);
DeviceEvent end_event(device);

// 记录开始事件
start_event.record();

// 执行一些操作
Tensor a = Tensor::empty({1000, 1000}, DataType::F32, device);
Tensor b = Tensor::empty({1000, 1000}, DataType::F32, device);
Tensor c = matmul(a, b);

// 记录结束事件
end_event.record();

// 等待结束事件完成
end_event.synchronize();

// 计算耗时
float elapsed_ms = start_event.elapsed_time(end_event);
std::cout << "Elapsed time: " << elapsed_ms << " ms" << std::endl;

// 查询事件状态
if (end_event.query()) {
    std::cout << "Event completed" << std::endl;
}
```

## 注意事项

- 两个事件必须在同一设备上才能测量耗时。
- 测量耗时前，两个事件都必须已被记录。
- 事件不支持拷贝，只支持移动构造和移动赋值。

## 相关链接

- [`context` 运行时上下文](../context/README.md)
- [`Tensor` 文档](../tensor/README.md)
