# `infinicore.context` 运行时上下文

运行时上下文模块，定义于 `InfiniCore/python/infinicore/context.py`，提供设备管理、流同步等运行时功能。

## 函数列表

- `get_device()`：获取当前活跃设备。
- `get_device_count(device_type)`：获取指定类型的设备数量。
- `set_device(device)`：设置当前活跃设备。
- `sync_stream()`：同步当前流。
- `sync_device()`：同步当前设备。
- `get_stream()`：获取当前流。

## 函数签名

```python
def get_device() -> device:
    """Get the current active device.
    
    Returns:
        device: The current active device object
    """

def get_device_count(device_type: str) -> int:
    """Get the number of available devices of a specific type.
    
    Args:
        device_type (str): The type of device to count (e.g., "cuda", "cpu", "npu")
    
    Returns:
        int: The number of available devices of the specified type
    """

def set_device(device) -> None:
    """Set the current active device.
    
    Args:
        device: The device to set as active
    """

def sync_stream() -> None:
    """Synchronize the current stream."""

def sync_device() -> None:
    """Synchronize the current device."""

def get_stream():
    """Get the current stream.
    
    Returns:
        stream: The current stream object
    """
```

## 示例

```python
import infinicore as ic

# 获取当前设备
current_device = ic.context.get_device()
print(f"Current device: {current_device}")

# 获取 CUDA 设备数量
cuda_count = ic.context.get_device_count("cuda")
print(f"CUDA devices: {cuda_count}")

# 设置设备
device = ic.device("cuda:0")
ic.context.set_device(device)

# 同步流和设备
ic.context.sync_stream()
ic.context.sync_device()

# 获取当前流
stream = ic.context.get_stream()
```

## 相关链接

- [`device` 文档](../device/README.md)
- [`Python API 总览`](../README.md)
