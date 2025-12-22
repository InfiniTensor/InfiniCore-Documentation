# `infinicore::op::attention`

解码阶段注意力算子，负责在 KV cache 中增量写入并返回当前 step 的输出。实现位于 `InfiniCore/src/infinicore/ops/attention/attention.cc`，头文件定义于 `InfiniCore/include/infinicore/ops/attention.hpp`。

## 函数签名

```cpp
namespace infinicore::op {
    Tensor attention(
        Tensor q,
        Tensor k,
        Tensor v,
        Tensor k_cache,
        Tensor v_cache,
        size_t pos
    );
    void attention_(
        Tensor out,
        Tensor q,
        Tensor k,
        Tensor v,
        Tensor k_cache,
        Tensor v_cache,
        size_t pos
    );
}
```

- `q`：查询张量，形状一般为 `(n_q_head, seq_len, head_dim)`。
- `k` / `v`：本 step 新增的 Key/Value，形状 `(n_kv_head, seq_len, head_dim)`。
- `k_cache` / `v_cache`：缓存张量，形状 `(n_kv_head, cache_len, head_dim)`，需保证 `pos + seq_len <= cache_len`。
- `pos`：写入位置索引（已填充 token 数）。
- `out`：输出张量（in-place 模式），需与输出形状 `(seq_len, n_q_head, head_dim)`、`dtype`、`device` 完全一致。

`attention` 函数创建新张量并返回；`attention_` 函数将结果写入 `out`。

## 行为说明

- 输入张量可为非连续布局，底层会自动处理。
- 支持分组 Query Attention（GQA），当 `n_q_head` 为 `n_kv_head` 的整数倍时自动映射。
- KV cache 在调用期间会写入 `[pos : pos + seq_len)` 区间，调用者需维护 `pos` 的累加。

## 示例

```cpp
#include <infinicore.hpp>

using namespace infinicore;

Device device(Device::Type::NVIDIA, 0);
context::setDevice(device);

Tensor q = Tensor::empty({8, 1, 128}, DataType::F16, device);
Tensor k = Tensor::empty({2, 1, 128}, DataType::F16, device);
Tensor v = Tensor::empty({2, 1, 128}, DataType::F16, device);
Tensor k_cache = Tensor::empty({2, 128, 128}, DataType::F16, device);
Tensor v_cache = Tensor::empty({2, 128, 128}, DataType::F16, device);

// Out-of-place
Tensor out = op::attention(q, k, v, k_cache, v_cache, 0);

// In-place
Tensor output = Tensor::empty({1, 8, 128}, DataType::F16, device);
op::attention_(output, q, k, v, k_cache, v_cache, 0);
```

## 相关链接

- [`infinicore::op` 索引](../README.md)
- [`nn` 模块](../../nn/README.md)
