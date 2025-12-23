# `infinicore.nn` 模块

`infinicore.nn` 聚合了面向神经网络的辅助模块，包括函数式算子、模块基类、参数管理等组件。

## 模块结构

| 子模块 | 说明 |
| --- | --- |
| `functional` | 函数式算子集合，接口文档见 [`functional/README.md`](functional/README.md)。 |
| `modules` | 模块容器类，包含 [`Module`](modules/module/README.md)、[`ModuleList`](modules/module_list/README.md)、[`Linear`](modules/linear/README.md)、[`RMSNorm`](modules/normalization/README.md)、[`RoPE`](modules/rope/README.md)、[`Embedding`](modules/sparse/README.md) 等。 |
| `Parameter` | 参数类，用于标识可训练参数，文档见 [`parameter/README.md`](parameter/README.md)。 |

## 使用示例

### 函数式算子

```python
import infinicore as ic
from infinicore.nn import functional as F

x = ic.ones((4, 1024), dtype=ic.float16, device=ic.device("cuda:0"))
normed = F.rms_norm(x, normalized_shape=[1024], weight=ic.ones((1024,), dtype=x.dtype, device=x.device))
activated = F.silu(normed)
```

### 模块化构建

```python
import infinicore as ic
from infinicore.nn import Module, ModuleList, Parameter

class Linear(Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = Parameter(ic.empty((out_features, in_features), dtype=ic.float16, device=ic.device("cuda:0")))
    
    def forward(self, x):
        return ic.matmul(x, self.weight.permute([1, 0]))

model = Linear(10, 5)
```

## 相关链接

- [`nn.functional 函数式文档`](functional/README.md)
- [`Python API 总览`](../README.md)
