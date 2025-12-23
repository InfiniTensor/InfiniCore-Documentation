# `infinicore.ops` 顶层算子

该模块通过 pybind11 将常用算子直接暴露在 `infinicore` 命名空间下，对应源码位于 `InfiniCore/python/infinicore/ops/`。所有函数均支持可选的 `out` 参数以复用输出缓冲区。

## 通用注意事项

- 所有参数必须是 `infinicore.Tensor` 实例（或至少携带 `_underlying` 指针），否则无法传递到底层 `_infinicore`。
- `out` 张量（若提供）需要和输出形状、dtype、设备完全匹配。
- 算子执行依赖底层 `_infinicore` 运行时；请在创建张量时传入期望的 `device`，保持所有输入处于同一设备上。

## 文档索引

- [`add`](add/README.md)
- [`matmul`](matmul/README.md)
- [`mul`](mul/README.md)
- [`narrow`](narrow/README.md)
- [`rearrange`](rearrange/README.md)
- [`attention`](attention/README.md)

## 示例

```python
import infinicore as ic

device = ic.device("cuda:0")
a = ic.ones((4, 8), dtype=ic.float16, device=device)
b = ic.ones((4, 8), dtype=ic.float16, device=device)

ic.add(a, b, out=a)  # 原位累加
c = ic.matmul(a, b.permute([1, 0]))  # (4, 8) @ (8, 4)

contiguous = ic.rearrange(c)

attn_out = ic.attention(
    q=contiguous,
    k=contiguous,
    v=contiguous,
    k_cache=ic.empty((4, 128, contiguous.shape[-1]), dtype=contiguous.dtype, device=contiguous.device),
    v_cache=ic.empty((4, 128, contiguous.shape[-1]), dtype=contiguous.dtype, device=contiguous.device),
    pos=0,
)
```

## 相关链接

- [`Python API 总览`](../README.md)
- [`nn.functional 函数式接口`](../nn/functional/README.md)
