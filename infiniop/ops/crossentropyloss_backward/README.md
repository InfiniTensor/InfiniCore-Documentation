# CrossEntropyLoss Backward 算子

CrossEntropyLoss Backward 算子用于计算交叉熵损失函数的反向传播梯度。该算子参考 `torch.nn.CrossEntropyLoss` 的反向传播行为。

## 数学定义

给定概率分布 $p$ 和目标标签 $t$，CrossEntropyLoss Backward 算子的计算公式为：

$$\frac{\partial L}{\partial \text{logits}_i} = p_i - \mathbf{1}_{t=i}$$

其中：
- $p_i$ 是经过 softmax 后的概率分布中第 $i$ 个类别的概率
- $\mathbf{1}_{t=i}$ 是指示函数，当目标标签 $t$ 等于 $i$ 时为 1，否则为 0
- $\frac{\partial L}{\partial \text{logits}_i}$ 是对 logits 第 $i$ 个元素的梯度

对于批量处理，每个样本的梯度计算相互独立。

## 接口

### 计算

```c
infiniStatus_t infiniopCrossEntropyLossBackward(
    infiniopCrossEntropyLossBackwardDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *grad_logits,
    const void *probs,
    const void *target,
    void *stream
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数：</div>

- `desc`:
  已使用 `infiniopCreateCrossEntropyLossBackwardDescriptor()` 初始化的算子描述符。
- `workspace`:
  额外工作空间地址。
- `workspace_size`:
  `workspace` 的大小，单位：字节。
- `grad_logits`:
  logits 梯度的计算结果地址，支持原位计算。
- `probs`:
  经过 softmax 后的概率分布张量数据指针。
- `target`:
  目标标签张量数据指针。
- `stream`:
  计算流/队列。

<div style="background-color: lightblue; padding: 1px;"> 返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_NULL_POINTER`], [`INFINI_STATUS_INSUFFICIENT_WORKSPACE`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`], [`INFINI_STATUS_INTERNAL_ERROR`].

### 创建算子描述

```c
infiniStatus_t infiniopCreateCrossEntropyLossBackwardDescriptor(
    infiniopHandle_t handle,
    infiniopCrossEntropyLossBackwardDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t grad_logits,
    infiniopTensorDescriptor_t probs,
    infiniopTensorDescriptor_t target
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数：</div>

- `handle`:
  硬件控柄。详见 [`InfiniopHandle_t`]。
- `desc_ptr`:
  `infiniopCrossEntropyLossBackwardDescriptor_t` 指针，指向将被初始化的算子描述符地址。
- `grad_logits` - { FP16 | FP32 | BF16 | (batch_size, num_classes) | (...) }:
  logits 梯度张量描述，支持原位计算。
- `probs` - { FP16 | FP32 | BF16 | (batch_size, num_classes) | (...) }:
  概率分布张量描述，支持原位计算。
- `target` - { INT32 | INT64 | (batch_size,) | (...) }:
  目标标签张量描述。

参数限制：

- `grad_logits` 和 `probs` 张量的数据类型必须相同，支持 `FP16`、`FP32`、`BF16`。
- `target` 张量的数据类型必须为 `INT32` 或 `INT64`。
- `grad_logits` 和 `probs` 张量的形状必须相同，通常为 `(batch_size, num_classes)`。
- `target` 张量的形状必须为 `(batch_size,)`，包含每个样本的目标类别索引。
- 支持原位计算，即计算时 `grad_logits` 可以和 `probs` 指向同一地址。

<div style="background-color: lightblue; padding: 1px;"> 返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_BAD_PARAM`], [`INFINI_STATUS_BAD_TENSOR_SHAPE`], [`INFINI_STATUS_BAD_TENSOR_DTYPE`], [`INFINI_STATUS_BAD_TENSOR_STRIDES`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`].

### 计算额外工作空间

```c
infiniStatus_t infiniopGetCrossEntropyLossBackwardWorkspaceSize(
    infiniopCrossEntropyLossBackwardDescriptor_t desc,
    size_t *size
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数：</div>

- `desc`:
  已使用 `infiniopCreateCrossEntropyLossBackwardDescriptor()` 初始化的算子描述符。
- `size`:
  额外空间大小的计算结果的写入地址。

<div style="background-color: lightblue; padding: 1px;"> 返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_NULL_POINTER`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`].

### 销毁算子描述符

```c
infiniStatus_t infiniopDestroyCrossEntropyLossBackwardDescriptor(
    infiniopCrossEntropyLossBackwardDescriptor_t desc
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数： </div>

- `desc`:
  待销毁的算子描述符。

<div style="background-color: lightblue; padding: 1px;"> 返回值： </div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`].

## 已知问题

无

<!-- 链接 -->
[`InfiniopHandle_t`]: /infiniop/handle/README.md

[`INFINI_STATUS_SUCCESS`]: /common/status/README.md#INFINI_STATUS_SUCCESS
[`INFINI_STATUS_BAD_PARAM`]: /common/status/README.md#INFINI_STATUS_BAD_PARAM
[`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`]: /common/status/README.md#INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED
[`INFINI_STATUS_BAD_TENSOR_SHAPE`]: /common/status/README.md#INFINI_STATUS_BAD_TENSOR_SHAPE
[`INFINI_STATUS_BAD_TENSOR_DTYPE`]: /common/status/README.md#INFINI_STATUS_BAD_TENSOR_DTYPE
[`INFINI_STATUS_BAD_TENSOR_STRIDES`]: /common/status/README.md#INFINI_STATUS_BAD_TENSOR_STRIDES
[`INFINI_STATUS_NULL_POINTER`]:/common/status/README.md#INFINI_STATUS_NULL_POINTER
[`INFINI_STATUS_INSUFFICIENT_WORKSPACE`]:/common/status/README.md#INFINI_STATUS_INSUFFICIENT_WORKSPACE
[`INFINI_STATUS_INTERNAL_ERROR`]:/common/status/README.md#INFINI_STATUS_INTERNAL_ERROR