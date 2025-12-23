# `infinicore.narrow`

窄化算子，在指定维度上缩小张量范围。定义于 `InfiniCore/python/infinicore/ops/narrow.py`。

## 函数签名

```python
def narrow(input: Tensor, dim: int, start: int, length: int) -> Tensor
```

- `input`：输入张量。
- `dim`：要窄化的维度索引。
- `start`：起始位置索引。
- `length`：要保留的长度。

返回一个共享底层存储的新张量，形状与输入相同，但 `dim` 维度的大小变为 `length`。

## 示例

```python
import infinicore as ic

device = ic.device("cuda:0")
a = ic.empty((10, 8), dtype=ic.float16, device=device)

# 在维度 0 上从索引 2 开始取 5 个元素
b = ic.narrow(a, dim=0, start=2, length=5)  # shape: (5, 8)

# 在维度 1 上从索引 1 开始取 4 个元素
c = ic.narrow(a, dim=1, start=1, length=4)  # shape: (10, 4)
```

## 注意事项

- 返回的张量与输入共享底层存储，修改返回的张量会影响原始张量。
- `start + length` 不能超过输入张量在 `dim` 维度的大小。

## 相关链接

- [`infinicore.ops` 索引](../README.md)
- [`Tensor` 视图操作](../../tensor/README.md)
