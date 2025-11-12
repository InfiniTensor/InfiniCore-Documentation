# `infinicore.nn` 模块

`infinicore.nn` 聚合了面向神经网络的辅助模块，目前主要暴露函数式算子集合 `infinicore.nn.functional`，并预留位置扩展其他组件（如模块化层、优化器等）。

## 模块结构

| 子模块 | 说明 |
| --- | --- |
| `functional` | 函数式算子集合，接口文档见 [`functional/README.md`](functional/README.md)。 |

## 使用示例

```python
import infinicore as ic
from infinicore.nn import functional as F

x = ic.ones((4, 1024), dtype=ic.float16, device=ic.device("cuda:0"))
normed = F.rms_norm(x, normalized_shape=[1024], weight=ic.ones((1024,), dtype=x.dtype, device=x.device))
activated = F.silu(normed)
```

## 相关链接

- [`nn.functional 函数式文档`](functional/README.md)
- [`Python API 总览`](../README.md)
