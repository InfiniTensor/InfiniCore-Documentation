# `nn.ModuleList`

模块列表容器，用于管理多个子模块。继承自 `Module`，支持索引访问和迭代。

## 类定义

```python
class infinicore.nn.ModuleList(modules: Optional[Sequence[Module]] = None)
```

## 主要功能

- **列表式访问**：支持索引、切片、迭代等 Python 列表操作。
- **自动注册**：添加到列表的模块会自动注册到父模块，可通过 `parameters()`、`state_dict()` 等方法访问。
- **类型安全**：仅接受 `Module` 及其子类实例。

## 常用方法

### `append(module: Module) -> ModuleList`

在列表末尾添加一个模块。

### `extend(modules: Sequence[Module]) -> ModuleList`

扩展列表，添加多个模块。

### `insert(index: int, module: Module) -> None`

在指定位置插入模块。

### `__getitem__(index: int | slice) -> Module | ModuleList`

支持整数索引和切片访问。

### `__len__() -> int`

返回模块数量。

### `__iter__() -> Iterator[Module]`

返回模块迭代器。

### `__iadd__(modules: Sequence[Module]) -> ModuleList`

支持 `+=` 操作符添加模块。

## 使用示例

```python
import infinicore as ic
from infinicore.nn import Module, ModuleList, Parameter

class MultiLayerModel(Module):
    def __init__(self, num_layers, hidden_size):
        super().__init__()
        self.layers = ModuleList([
            Linear(hidden_size, hidden_size) for _ in range(num_layers)
        ])
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
```

## 相关链接

- [`Module`](module/README.md)
- [`nn.modules` 概览](../README.md)


