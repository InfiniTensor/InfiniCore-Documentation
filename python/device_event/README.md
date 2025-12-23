# `infinicore.DeviceEvent`

设备事件类，定义于 `InfiniCore/python/infinicore/device_event.py`，用于同步操作和性能测量。

## 概述

`DeviceEvent` 类似于 `torch.cuda.Event`，提供以下功能：
- 在特定设备流上记录事件
- 与事件同步
- 测量事件之间的耗时
- 查询事件完成状态
- 让流等待事件

## 构造函数

```python
DeviceEvent(enable_timing=False, device=None)
```

- `enable_timing`：是否记录计时数据，默认为 `False`。
- `device`：目标设备，如果为 `None`，使用当前设备。

## 主要方法

- `record(stream=None)`：在流上记录事件。如果 `stream` 为 `None`，使用当前流。
- `synchronize()`：等待事件完成（阻塞）。
- `query()`：检查事件是否已完成，返回 `bool`。
- `elapsed_time(other)`：计算与另一个事件之间的耗时（毫秒）。
- `wait(stream=None)`：让指定流等待此事件。

## 属性

- `device`：事件所在的设备。
- `is_recorded`：事件是否已被记录。
- `enable_timing`：是否记录计时数据。

## 示例

```python
import infinicore as ic

device = ic.device("cuda:0")
ic.context.set_device(device)

# 创建事件
start_event = ic.DeviceEvent(enable_timing=True, device=device)
end_event = ic.DeviceEvent(enable_timing=True, device=device)

# 记录开始事件
start_event.record()

# 执行一些操作
a = ic.empty((1000, 1000), dtype=ic.float32, device=device)
b = ic.empty((1000, 1000), dtype=ic.float32, device=device)
c = ic.matmul(a, b)

# 记录结束事件
end_event.record()

# 等待结束事件完成
end_event.synchronize()

# 计算耗时
elapsed_ms = start_event.elapsed_time(end_event)
print(f"Elapsed time: {elapsed_ms} ms")

# 查询事件状态
if end_event.query():
    print("Event completed")
```

## 注意事项

- 两个事件必须在同一设备上才能测量耗时。
- 测量耗时前，两个事件都必须已启用计时（`enable_timing=True`）且已被记录。
- 事件不支持拷贝，只支持移动语义。

## 相关链接

- [`context` 运行时上下文](../context/README.md)
- [`Tensor` 文档](../tensor/README.md)
