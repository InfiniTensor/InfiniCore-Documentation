# `nn.Parameter`

一种特殊的 `Tensor`，用于表示模块的可训练参数。

## 类定义

```python
class infinicore.nn.Parameter(data: Tensor)
```

## 主要功能

- **参数标识**：继承自 `Tensor`，但被 `Module` 识别为可训练参数。
- **自动注册**：当作为 `Module` 的属性赋值时，会被自动注册到 `_parameters` 字典。
- **状态字典**：参数会被包含在 `state_dict()` 中，用于模型保存与加载。

## 约束

- `data` 必须是 `infinicore.Tensor` 实例。
- 不支持深拷贝（`__deepcopy__`）和序列化（`__reduce_ex__`）。

## 使用示例

```python
import infinicore as ic
from infinicore.nn import Module, Parameter

class Linear(Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        device = ic.device("cuda:0")
        self.weight = Parameter(ic.empty((out_features, in_features), dtype=ic.float16, device=device))
        self.bias = Parameter(ic.empty((out_features,), dtype=ic.float16, device=device))
```

## 相关链接

- [`Module`](../modules/module/README.md)
- [`nn` 模块概览](../README.md)
