# `infinicore.tensor` 模块

`Tensor` 类及核心构造函数定义在 `InfiniCore/python/infinicore/tensor.py`，负责在 Python 端封装底层 `_infinicore.Tensor` 指针并提供常用操作。

## `Tensor` 类

### 主要属性

- `shape`（`tuple[int, ...]`）/ `ndim` / `size(dim)`：获取张量维度信息。
- `stride(dim=None)`：返回步长数组或指定维度的步长。
- `dtype`：对应的标量类型（参见 [`dtype` 文档](../dtype/README.md)）。
- `device`：张量所在设备（参见 [`device` 文档](../device/README.md)）。
- `numel()`：元素总数。
- `is_contiguous()`：判断是否连续存储。

### 常用方法

- `copy_(src)`：将 `src` 的数据拷贝到当前张量。
- `to(*args, **kwargs)`：执行跨设备/数据类型转换，返回新 `Tensor`。
- `as_strided(size, stride)`：创建共享存储的视图。
- `contiguous()`：返回连续副本。
- `permute(dims)`：重新排列维度。
- `view(shape)`：改变张量形状（需满足可视条件）。
- `debug(filename=None)`：打印张量信息或将原始数据写入文件。

## 构造函数

全部为 `Tensor` 类的顶层函数，默认要求显式传入 `dtype` 与 `device`：

```python
from infinicore import (
    empty, strided_empty, zeros, ones,
    from_blob, strided_from_blob, empty_like,
)
```

### 说明

- `empty(shape, *, dtype, device, pin_memory=False)`：按给定形状分配未初始化存储。
- `strided_empty(shape, strides, *, dtype, device, pin_memory=False)`：按指定步长分配存储。
- `zeros` / `ones`：与 `empty` 类似，但在 Python 层暂时未初始化填充值。
- `from_blob(data_ptr, shape, *, dtype, device)`：将外部内存包装为 `Tensor`，不接管内存所有权。
- `strided_from_blob`：同上但可显式指定步长。
- `empty_like(input, *, dtype=None, device=None)`：按照 `input` 的形状/步长创建新张量，可覆盖 `dtype` 或 `device`。

## 示例

```python
import infinicore as ic

device = ic.device("cuda:0")
a = ic.empty((4, 8), dtype=ic.float16, device=device)
b = ic.ones((4, 8), dtype=ic.float16, device=device)

a.copy_(b)
c = a.permute([1, 0]).contiguous()
```

## 相关链接

- [`infinicore` 模块概览](../README.md)
- [`顶层算子`](../ops/README.md)
- [`nn.functional`](../nn/functional/README.md)
