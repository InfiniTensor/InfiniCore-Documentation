# `nn.Module`

所有神经网络模块的基类，提供参数注册、状态字典管理、设备迁移等核心功能。

## 类定义

```python
class infinicore.nn.Module
```

## 主要功能

- **参数管理**：通过 `register_parameter()` 注册可训练参数，自动追踪到 `parameters()` 迭代器。
- **缓冲区管理**：通过 `register_buffer()` 注册非参数状态（如 BatchNorm 的 running_mean），可选择是否持久化到状态字典。
- **子模块管理**：支持嵌套模块结构，可通过 `add_module()` 添加子模块。
- **状态字典**：`state_dict()` 与 `load_state_dict()` 用于模型保存与加载。
- **设备迁移**：`to()` 方法支持将模块及其参数/缓冲区迁移到指定设备。

## 常用方法

### `register_parameter(name: str, param: Optional[Parameter]) -> None`

注册一个参数到模块。参数会被添加到 `_parameters` 字典，并可通过属性访问。

### `register_buffer(name: str, tensor: Optional[Tensor], persistent: bool = True) -> None`

注册一个缓冲区。若 `persistent=False`，该缓冲区不会保存到状态字典中。

### `add_module(name: str, module: Optional[Module]) -> None`

添加一个子模块。子模块会被添加到 `_modules` 字典，并可通过属性访问。

### `parameters(recurse: bool = True) -> Iterator[Parameter]`

返回模块所有参数的迭代器。`recurse=True` 时会递归包含子模块的参数。

### `buffers(recurse: bool = True, remove_duplicate: bool = True) -> Iterator[Tensor]`

返回模块所有缓冲区的迭代器。

### `state_dict(destination=None, prefix='', keep_vars=False) -> Dict[str, Tensor]`

返回模块的状态字典，包含所有参数和持久化缓冲区。

### `load_state_dict(state_dict: Mapping[str, Tensor], strict: bool = True) -> _IncompatibleKeys`

从状态字典加载参数和缓冲区。返回 `_IncompatibleKeys` 对象，包含缺失和意外的键。

### `to(device) -> Module`

将模块及其参数/缓冲区迁移到指定设备。`device` 可以是 `infinicore.device` 实例或字符串（如 `"cuda:0"`）。

### `train(mode: bool = True) -> Module`

设置模块的训练/评估模式。当前实现为占位，保留接口以兼容 PyTorch 风格。

### `eval() -> Module`

将模块设置为评估模式（等价于 `train(False)`）。

## 使用示例

```python
import infinicore as ic
from infinicore.nn import Module, Parameter

class Linear(Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = Parameter(ic.empty((out_features, in_features), dtype=ic.float16, device=ic.device("cuda:0")))
        self.bias = Parameter(ic.empty((out_features,), dtype=ic.float16, device=ic.device("cuda:0")))
    
    def forward(self, x):
        return ic.matmul(x, self.weight.permute([1, 0])) + self.bias

model = Linear(10, 5)
state = model.state_dict()
model.load_state_dict(state)
```

## 相关链接

- [`ModuleList`](module_list/README.md)
- [`Parameter`](../../parameter/README.md)
- [`nn` 模块概览](../../README.md)


