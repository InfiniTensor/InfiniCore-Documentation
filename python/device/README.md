# `infinicore.device`

设备句柄类，定义于 `InfiniCore/python/infinicore/device.py`，用于在 Python 端选择和描述运行时设备。

## 构造方式

```python
from infinicore import device

cpu = device()                 # 默认 "cpu"
cuda0 = device("cuda:0")       # 字符串形式指定
cuda1 = device("cuda", 1)      # 类型 + index
clone = device(cuda0)          # 从已有实例拷贝
```

- `type`：支持 `"cpu"`、`"cuda"`、`"mlu"`、`"npu"`、`"musa"` 等，具体取决于 `_infinicore` 编译时支持。
- `index`：可选整型索引，字符串中已包含 `":"` 时禁止再传入。
- 传入已有 `device` 实例时会拷贝其 `type`/`index`。

## 属性与方法

- `type` / `index`：公开的设备类型与序号。
- `__repr__()` / `__str__()`：打印友好格式，如 `device(type='cuda', index=0)` 或 `"cuda:0"`。
- `_underlying`：内部 `_infinicore.Device` 对象，供底层 API 使用。

## 与运行时的关系

- `device` 实例用于所有张量构造函数及顶层算子，确保 `_infinicore` 在正确的设备上执行。
- 若需要从底层 `_infinicore.Device` 转换为 Python 对象，可使用 `device._from_infinicore_device`（内部方法）。

## 相关链接

- [`Tensor` 构造函数](../tensor/README.md#构造函数)
- [`运行时上下文`](../README.md#运行时上下文)

