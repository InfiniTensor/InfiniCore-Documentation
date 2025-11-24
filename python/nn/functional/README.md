# `infinicore.nn.functional` 函数式接口

`infinicore.nn.functional` 集中收录 PyTorch 风格的函数式算子封装。实现位于 `InfiniCore/python/infinicore/nn/functional.py`，依赖 `_infinicore` C++ 绑定并复用运行时上下文。

## 公共约定

- 所有函数都返回 `infinicore.Tensor`；当提供 `out`/`inplace` 等参数时会复用已有缓冲区。
- 输入张量需由 `infinicore` 创建（或至少携带 `_underlying` 指针），否则无法与底层运行时交互。
- 若函数内部调用 `_infinicore.*_` 原位接口，需确保输出张量与输入形状、dtype 一致。

## API 详情

- [`causal_softmax`](causal_softmax/README.md)：因果掩码 Softmax。
- [`rms_norm`](rms_norm/README.md)：Root Mean Square LayerNorm。
- [`silu`](silu/README.md)：SiLU（Sigmoid Linear Unit）激活。
- [`swiglu`](swiglu/README.md)：SwiGLU 前向门控。

## 示例

```python
import infinicore as ic
from infinicore.nn import functional as F

x = ic.empty((4, 1024), dtype=ic.float16, device=ic.device("cuda:0"))
w = ic.empty((1024,), dtype=ic.float16, device=x.device)

normed = F.rms_norm(x, normalized_shape=list(w.shape), weight=w)
activated = F.silu(normed)
gated = F.swiglu(activated, ic.empty_like(activated))

probs = F.causal_softmax(gated, out=ic.empty_like(gated))
```

## 相关链接

- [`Python API 总览`](../../README.md)
- [`ntops` 协作接口说明](../../README.md#与-ntops-的协作)
