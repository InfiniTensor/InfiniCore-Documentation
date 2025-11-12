# `infinicore.attention`

解码阶段注意力算子，负责在 KV cache 中增量写入并返回当前 step 的输出。实现位置：`InfiniCore/python/infinicore/ops/attention.py`。

## 函数签名

```python
def attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    k_cache: Tensor,
    v_cache: Tensor,
    pos: int,
    *,
    out: Optional[Tensor] = None,
) -> Tensor
```

- `q`：查询张量，形状一般为 `(n_q_head, seq_len, head_dim)`。
- `k` / `v`：本 step 新增的 Key/Value，形状 `(n_kv_head, seq_len, head_dim)`。
- `k_cache` / `v_cache`：缓存张量，形状 `(n_kv_head, cache_len, head_dim)`，需保证 `pos + seq_len <= cache_len`。
- `pos`：写入位置索引（已填充 token 数）。
- `out`：可选输出张量，若提供需与输出形状 `(seq_len, n_q_head, head_dim)`、`dtype`、`device` 完全一致。

默认情况下函数会创建新张量并返回；提供 `out` 时调用 `_infinicore.attention_` 原位写入。

## 行为说明

- 输入张量可为非连续布局，底层会自动处理。
- 支持分组 Query Attention（GQA），当 `n_q_head` 为 `n_kv_head` 的整数倍时自动映射。
- KV cache 在调用期间会写入 `[pos : pos + seq_len)` 区间，调用者需维护 `pos` 的累加。

## 示例

```python
import infinicore as ic

device = ic.device("cuda:0")

q = ic.empty((8, 1, 128), dtype=ic.float16, device=device)
k = ic.empty((2, 1, 128), dtype=ic.float16, device=device)
v = ic.empty((2, 1, 128), dtype=ic.float16, device=device)
k_cache = ic.empty((2, 128, 128), dtype=ic.float16, device=device)
v_cache = ic.empty((2, 128, 128), dtype=ic.float16, device=device)

out = ic.attention(q, k, v, k_cache, v_cache, pos=0)
```

## 相关链接

- [`infinicore.ops` 索引](../README.md)
- [`nn.functional` 文档](../../nn/functional/README.md)
