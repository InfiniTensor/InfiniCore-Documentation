# `infinicore.matmul`

矩阵乘法/GEMM 前端，封装 `InfiniCore/python/infinicore/ops/matmul.py` 中的 pybind11 绑定。

## 函数签名

```python
def matmul(input: Tensor, other: Tensor, *, out: Optional[Tensor] = None) -> Tensor
```

- `input`：左乘矩阵，形状需满足 GEMM 要求，可包含批维。
- `other`：右乘矩阵，与 `input` 的维度兼容。
- `out`：可选输出张量；若提供需与结果形状、`dtype`、`device` 完全一致。

默认返回新张量；当提供 `out` 时调用 `_infinicore.matmul_` 原地写入。底层会复用 *InfiniOP* GEMM 描述符完成计算。

## 输入要求

- 支持常见数据类型（如 `float16`、`float32`、`bfloat16`）。
- 支持批量维度；所有批维需两输入对齐。
- 当 `out` 与输入不处于同一设备或数据类型不匹配时，底层会抛出异常。

## 示例

```python
import infinicore as ic

device = ic.device("cuda:0")
a = ic.ones((4, 8), dtype=ic.float16, device=device)
b = ic.ones((8, 16), dtype=ic.float16, device=device)

c = ic.matmul(a, b)                 # 创建新张量
ic.matmul(a, b, out=c)              # 原位复用输出缓冲
```

## 相关链接

- [`infinicore.ops` 索引](../README.md)
- [`Tensor` 构造函数](../../README.md#张量与构造函数)
