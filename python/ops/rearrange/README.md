# `infinicore.rearrange`

调整张量布局或生成连续副本的辅助算子，定义于 `InfiniCore/python/infinicore/ops/rearrange.py`。

## 函数签名

```python
def rearrange(input: Tensor, other: Optional[Tensor] = None, *, out: Optional[Tensor] = None) -> Tensor
```

- `input`：源张量。
- `other`：当前保留未使用，作为后续扩展占位。
- `out`：可选输出张量，若提供需与 `input` 形状、`dtype`、`device` 一致。

默认返回新张量；当提供 `out` 时调用 `_infinicore.rearrange_` 将结果写入既有缓冲区。

## 常见用途

- 将任意步长的张量转换为连续布局。
- 为跨设备拷贝或下游算子准备符合要求的存储格式。
- 与 `Tensor.contiguous()` 行为一致，但可选择写入指定输出。

## 示例

```python
import infinicore as ic

device = ic.device("cuda:0")
x = ic.empty((4, 8), dtype=ic.float16, device=device)

y = ic.rearrange(x)      # 返回连续副本
ic.rearrange(x, out=x)   # 原位整理（若底层支持）
```

## 相关链接

- [`infinicore.ops` 索引](../README.md)
- [`Tensor` 常用方法](../../README.md#张量与构造函数)
