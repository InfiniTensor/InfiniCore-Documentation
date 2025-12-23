# `infinicore.nn.functional.random_sample`

随机采样函数，从 logits 中采样索引，支持 nucleus/top-k 过滤。定义于 `InfiniCore/python/infinicore/nn/functional/random_sample.py`。

## 函数签名

```python
def random_sample(
    logits: Tensor,
    random_val: float,
    topp: float,
    topk: int,
    temperature: float,
    *,
    out=None,
) -> Tensor
```

## 参数说明

- `logits`：输入 logits 张量，形状为 `(vocab_size,)` 或 `(batch_size, vocab_size)`。
- `random_val`：随机值，用于采样。
- `topp`：nucleus sampling 的阈值（top-p），范围 [0, 1]。
- `topk`：top-k sampling 的 k 值。
- `temperature`：温度参数，用于控制采样的随机性。
- `out`：可选输出张量，若提供需与结果形状、`dtype`、`device` 一致。

## 返回值

返回采样得到的索引张量：
- 如果输入形状为 `(vocab_size,)`，返回标量张量。
- 如果输入形状为 `(batch_size, vocab_size)`，返回形状为 `(batch_size,)` 的张量。

## 示例

```python
import infinicore as ic
from infinicore.nn import functional as F

device = ic.device("cuda:0")

# 创建 logits
logits = ic.empty((10000,), dtype=ic.float32, device=device)
# ... 填充 logits 值 ...

# 随机采样
import random
random_val = random.random()
sampled_idx = F.random_sample(
    logits,
    random_val=random_val,
    topp=0.9,      # nucleus sampling threshold
    topk=50,       # top-k sampling
    temperature=1.0
)

# 批量采样
batch_logits = ic.empty((4, 10000), dtype=ic.float32, device=device)
batch_indices = F.random_sample(
    batch_logits,
    random_val=random.random(),
    topp=0.9,
    topk=50,
    temperature=0.8
)  # shape: (4,)
```

## 相关链接

- [`nn.functional` 函数式接口](../README.md)
