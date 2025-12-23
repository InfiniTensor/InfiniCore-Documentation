# `infinicore.utils` 工具函数

工具函数模块，定义于 `InfiniCore/python/infinicore/utils.py`，提供数据类型转换等工具函数。

## 函数列表

### 数据类型转换

- `to_torch_dtype(infini_dtype)`：将 infinicore 数据类型转换为 PyTorch 数据类型。
- `to_infinicore_dtype(torch_dtype)`：将 PyTorch 数据类型转换为 infinicore 数据类型。
- `numpy_to_infinicore_dtype(numpy_dtype)`：将 NumPy 数据类型转换为 infinicore 数据类型。
- `infinicore_to_numpy_dtype(infini_dtype)`：将 infinicore 数据类型转换为 NumPy 数据类型。

## 函数签名

```python
def to_torch_dtype(infini_dtype) -> torch.dtype:
    """Convert infinicore data type to PyTorch data type"""

def to_infinicore_dtype(torch_dtype) -> dtype:
    """Convert PyTorch data type to infinicore data type"""

def numpy_to_infinicore_dtype(numpy_dtype) -> dtype:
    """Convert numpy data type to infinicore data type"""

def infinicore_to_numpy_dtype(infini_dtype) -> numpy.dtype:
    """Convert infinicore data type to numpy data type"""
```

## 支持的数据类型

### infinicore ↔ PyTorch

- `float16` ↔ `torch.float16`
- `float32` ↔ `torch.float32`
- `bfloat16` ↔ `torch.bfloat16`
- `int8` ↔ `torch.int8`
- `int16` ↔ `torch.int16`
- `int32` ↔ `torch.int32`
- `int64` ↔ `torch.int64`
- `uint8` ↔ `torch.uint8`

### infinicore ↔ NumPy

- `float32` ↔ `np.float32`
- `float64` ↔ `np.float64`
- `float16` ↔ `np.float16`
- `bfloat16` ↔ `ml_dtypes.bfloat16`
- `int8` ↔ `np.int8`
- `int16` ↔ `np.int16`
- `int32` ↔ `np.int32`
- `int64` ↔ `np.int64`
- `uint8` ↔ `np.uint8`

## 示例

```python
import infinicore as ic
import infinicore.utils as utils
import torch
import numpy as np

# infinicore ↔ PyTorch
ic_dtype = ic.float16
torch_dtype = utils.to_torch_dtype(ic_dtype)  # torch.float16
ic_dtype_back = utils.to_infinicore_dtype(torch_dtype)  # ic.float16

# infinicore ↔ NumPy
numpy_dtype = utils.infinicore_to_numpy_dtype(ic.float32)  # np.float32
ic_dtype_back = utils.numpy_to_infinicore_dtype(numpy_dtype)  # ic.float32
```

## 相关链接

- [`dtype` 文档](../dtype/README.md)
- [`Python API 总览`](../README.md)
