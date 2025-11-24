# `infinicore.dtype` 与标量类型

`InfiniCore/python/infinicore/dtype.py` 导出与 C++ 端一致的标量类型枚举。通过 `from infinicore import dtype, float16, int32, ...` 可直接访问。

## 常用类型列表

- `dtype`：枚举工厂，可用于创建自定义 dtype 或从 `_underlying` 还原。
- 浮点类型：`float`, `float16`, `float32`, `float64`, `half`, `bfloat16`, `double`.
- 复数类型：`cfloat`, `cdouble`, `complex32`, `complex64`, `complex128`.
- 整型：`int`, `int8`, `int16`, `int32`, `int64`, `short`, `long`, `uint8`.
- 布尔：`bool`.

所有类型对象都暴露 `_underlying` 属性，用于在调用底层 `_infinicore` 接口时传递。

## 示例

```python
import infinicore as ic

cpu = ic.device("cpu")
a = ic.empty((4, 8), dtype=ic.float16, device=cpu)

if a.dtype is ic.float16:
    print("half precision tensor")
```

## 相关链接

- [`Tensor` 构造函数](../tensor/README.md#构造函数)
- [`infinicore` 模块概览](../README.md)
