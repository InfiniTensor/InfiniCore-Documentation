# `infinicore.add`

逐元素加法算子，支持广播与非连续张量。定义于 `InfiniCore/python/infinicore/ops/add.py`。

## 函数签名

```python
def add(input: Tensor, other: Tensor, *, out: Optional[Tensor] = None) -> Tensor
```

- `input`：左操作数张量。
- `other`：右操作数张量，可与 `input` 形状相同或可广播到 `input`。
- `out`：可选输出张量，若提供需与结果形状、`dtype`、`device` 一致。

若未提供 `out`，函数会创建并返回新张量；提供 `out` 时将调用 `_infinicore.add_` 在原地写入。

## 示例

```python
import infinicore as ic

device = ic.device("cuda:0")
a = ic.ones((4, 8), dtype=ic.float16, device=device)
b = ic.ones((1, 8), dtype=ic.float16, device=device)  # 可广播

out = ic.add(a, b)        # 返回新张量
ic.add(a, b, out=a)       # 原位累加
```

## 相关链接

- [`infinicore.ops` 索引](../README.md)
- [`Tensor` 构造函数](../../README.md#张量与构造函数)
