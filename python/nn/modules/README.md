# `infinicore.nn.modules` 模块容器

`infinicore.nn.modules` 提供了神经网络模块的基础类和容器类，用于构建可组合的模型结构。

## 模块列表

- [`Module`](module/README.md)：所有神经网络模块的基类，提供参数管理、状态字典等功能。
- [`ModuleList`](module_list/README.md)：模块列表容器，用于管理多个子模块。

## 使用示例

```python
import infinicore as ic
from infinicore.nn import Module, ModuleList, Parameter

class MyModel(Module):
    def __init__(self):
        super().__init__()
        self.layers = ModuleList([
            # 子模块列表
        ])
        self.weight = Parameter(ic.empty((10, 10), dtype=ic.float16, device=ic.device("cuda:0")))
```

## 相关链接

- [`nn` 模块概览](../README.md)
- [`Parameter`](../parameter/README.md)


