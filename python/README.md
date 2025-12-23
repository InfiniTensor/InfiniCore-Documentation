# `infinicore` Python 前端

*InfiniCore* 提供了与 C++ 前端一致的 Python 封装，位于 `python/infinicore/`。该模块通过 `pybind11` 将核心张量、算子与设备上下文暴露给 Python，便于在推理框架或调试脚本中快速集成。

## 模块结构

| 符号 | 说明 |
| --- | --- |
| `device` | 设备句柄类 (`python/infinicore/device.py`)，支持 `"cuda:0"`、`device(\"cpu\", 0)` 等写法或复用已有实例。 |
| `dtype` / `float16` 等 | 数据类型枚举 (`python/infinicore/dtype.py`)。 |
| `Tensor` | 张量包装类 (`python/infinicore/tensor.py`)，内部封装底层 `_infinicore` 对象。 |
| `empty` / `zeros` / `ones` / `empty_like` 等 | 张量构造函数（`python/infinicore/tensor.py`），默认要求显式传入 `dtype` 与 `device`。 |
| `context` | 运行时上下文模块 (`python/infinicore/context.py`)，提供设备管理、流同步等功能。 |
| `DeviceEvent` | 设备事件类 (`python/infinicore/device_event.py`)，用于同步和性能测量。 |
| `utils` | 工具函数模块 (`python/infinicore/utils.py`)，提供数据类型转换等功能。 |
| 顶层算子（`add`、`matmul`、`mul`、`narrow`、`rearrange`、`attention`） | 暴露在 `infinicore` 命名空间下，对应实现位于 `python/infinicore/ops/`。 |
| `infinicore.nn` | 神经网络相关模块集合，包括 `Linear`、`RMSNorm`、`RoPE`、`Embedding` 等。 |
| `infinicore.nn.functional` | 函数式算子集合 (`python/infinicore/nn/functional/`)，包括 `rope`、`random_sample`、`linear`、`embedding` 等。 |
| `use_ntops` / `infinicore.ntops` | 若系统安装 `ntops` 包，将自动置位 `use_ntops=True` 并暴露原始模块。 |

所有符号在包的 `__init__.py` 中进行了显式导出，可直接通过 `import infinicore as ic` 后使用。

相关导出定义见 `InfiniCore/python/infinicore/__init__.py`。

## API 索引

- [`device`](device/README.md)
- [`dtype`](dtype/README.md)
- [`Tensor`](tensor/README.md)
- [`empty`](tensor/README.md#构造函数)
- [`zeros`](tensor/README.md#构造函数)
- [`ones`](tensor/README.md#构造函数)
- [`empty_like`](tensor/README.md#构造函数)
- [`from_blob`](tensor/README.md#构造函数)
- [`context`](context/README.md)
- [`DeviceEvent`](device_event/README.md)
- [`utils`](utils/README.md)
- [`add`](ops/add/README.md)
- [`matmul`](ops/matmul/README.md)
- [`mul`](ops/mul/README.md)
- [`narrow`](ops/narrow/README.md)
- [`rearrange`](ops/rearrange/README.md)
- [`attention`](ops/attention/README.md)
- [`nn`](nn/README.md)
- [`nn.Module`](nn/modules/module/README.md)
- [`nn.ModuleList`](nn/modules/module_list/README.md)
- [`nn.Linear`](nn/modules/linear/README.md)
- [`nn.RMSNorm`](nn/modules/normalization/README.md)
- [`nn.RoPE`](nn/modules/rope/README.md)
- [`nn.Embedding`](nn/modules/sparse/README.md)
- [`nn.Parameter`](nn/parameter/README.md)
- [`nn.functional`](nn/functional/README.md)
- [`nn.functional.causal_softmax`](nn/functional/causal_softmax/README.md)
- [`nn.functional.rms_norm`](nn/functional/rms_norm/README.md)
- [`nn.functional.silu`](nn/functional/silu/README.md)
- [`nn.functional.swiglu`](nn/functional/swiglu/README.md)
- [`nn.functional.rope`](nn/functional/rope/README.md)
- [`nn.functional.random_sample`](nn/functional/random_sample/README.md)
- [`nn.functional.linear`](nn/functional/linear/README.md)
- [`nn.functional.embedding`](nn/functional/embedding/README.md)

## 张量与构造函数

`Tensor` 是对底层 `_infinicore.Tensor` 的 Python 包装，常用接口包括：

- `shape` / `ndim` / `size(dim)` / `stride(dim)`：获取张量维度与步长信息。
- `dtype` / `device`：返回 `dtype` 与 `device` 包装类。
- `numel()` / `is_contiguous()`：查看张量元素数量与存储布局。
- `copy_(src)` / `to(...)`：执行数据拷贝与跨设备搬运。
- `contiguous()` / `permute(dims)` / `view(shape)` / `as_strided(size, stride)`：布局调整与视图操作。
- `debug(filename=None)`：将张量内容打印或输出到二进制文件。

常用构造函数包括 `empty`、`strided_empty`、`zeros`、`ones`、`from_blob`、`strided_from_blob`、`empty_like` 等：

```python
import infinicore as ic

cpu = ic.device("cpu")
a = ic.empty((4, 8), dtype=ic.float16, device=cpu)
b = ic.ones((4, 8), dtype=ic.float16, device=cpu)
a.copy_(b)
```

> 注意：这些函数要求显式传入 `dtype` 与 `device`，避免隐式从 PyTorch/TensorFlow 对象推断。

## 顶层算子 (`infinicore.*`)

详见 [`ops` 文档索引`](ops/README.md) 及各算子文档。

## 函数式算子 (`infinicore.nn.functional`)

详见 [`nn.functional` 文档](nn/functional/README.md) 及子目录。

## 运行时上下文

- `_infinicore` 在进程内维护运行时状态；创建张量时请显式传入 `device`，并保持算子的所有输入位于同一设备。
- 如需强制同步，可调用 `infinicore.lib._infinicore.sync_stream()`、`sync_device()` 等底层绑定。
- 在同一执行流内串行调用算子通常无需额外同步。

## 与 `ntops` 的协作

- 导入 `ntops` 成功后，`infinicore.use_ntops` 会被设置为 `True`，并可通过 `infinicore.ntops` 访问原始模块。
- `nn.functional.silu` 在 `use_ntops=True` 且设备类型为 `"cuda"`/`"musa"` 且未传 `out` 时，会委托 `ntops.torch.silu`。
- 若想强制禁用，可直接设置 `infinicore.use_ntops = False`。

## 端到端示例

```python
import infinicore as ic

device = ic.device("cuda:0")

q = ic.empty((8, 1, 128), dtype=ic.float16, device=device)
k = ic.empty((2, 1, 128), dtype=ic.float16, device=device)
v = ic.empty((2, 1, 128), dtype=ic.float16, device=device)
k_cache = ic.empty((2, 128, 128), dtype=ic.float16, device=device)
v_cache = ic.empty((2, 128, 128), dtype=ic.float16, device=device)

out = ic.attention(q, k, v, k_cache, v_cache, pos=0)

if ic.use_ntops:
    # 在部分设备上，SiLU 会委托给 ntops 的高性能实现
    out = ic.nn.functional.silu(out)

out.debug()
```

## 相关链接

- [`infinicore.ops 顶层算子`](ops/README.md)
- [`nn.functional 函数式文档`](nn/functional/README.md)
- [`InfiniOP` 统一算子库](/infiniop/README.md)
