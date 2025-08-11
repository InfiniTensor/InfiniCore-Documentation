
# `CrossEntropyLossBackward`

`CrossEntropyLossBackward`, 即**交叉熵损失函数反向**算子，为双目逐元素算子。其计算可被表述为：

```math
grad\_logits = \frac{(probs - target)}{N}
```

其中 `probs` 和 `target` 为输入，`grad_logits` 为输出，`N`为batch_size。
- `probs`: 形状为[batch_size, num_classes]，表示softmax输出的概率分布$(\left( \sum = 1.0 \right))$。
- `target`：形状为[batch_size, num_classes]，表示真实标签的one-hot编码(仅一个位置为1)。
- `grad_logits`: 形状为[batch_size, num_classes]，表示损失函数对原始logits的梯度。
> 若输入形状大于2维，则最低维为num_classes，其余维度乘积为batch_size，即是`N`。

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

<div style="background-color: lightblue; padding: 1px;"> 参数： </div>

- `desc`:
  已使用 `infiniopCreateCrossEntropyLossBackwardDescriptor()` 初始化的算子描述符；
- `workspace`:
  指向算子计算所需的额外工作空间；
- `workspace_size`:
  `workspace` 的大小，单位：字节；
- `grad_logits`:
  输出张量。张量限制见[创建算子描述](#创建算子描述)部分；
- `probs`:
  输入张量。张量限制见[创建算子描述](#创建算子描述)部分；
- `target`:
  输入张量。张量限制见[创建算子描述](#创建算子描述)部分；
- `stream`:
  计算流/队列；

<div style="background-color: lightblue; padding: 1px;"> 返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_BAD_PARAM`], [`INFINI_STATUS_INSUFFICIENT_WORKSPACE`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`], [`INFINI_STATUS_INTERNAL_ERROR`]，[`INFINI_STATUS_BAD_TENSOR_DTYPE`].

### 创建算子描述

```c
infiniStatus_t infiniopCreateCrossEntropyLossBackwardDescriptor(
    infiniopHandle_t handle,
    infiniopCrossEntropyLossBackwardDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t grad_logits_desc,
    infiniopTensorDescriptor_t probs_desc,
    infiniopTensorDescriptor_t target_desc
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数：</div>

- `handle`:
  `infiniopHandle_t` 类型的硬件控柄。详情请看：[`InfiniopHandle_t`]。
- `desc_ptr`:
  `infiniopCrossEntropyLossBackwardDescriptor_t` 指针，指向将被初始化的算子描述符地址；
- `grad_logits_desc` - { dT | (d1,...,dn) | (...) }:
  算子计算参数 `grad_logits` 的张量描述，支持原位计算。
- `probs_desc` - { dT | (d1,...,dn) | (...) }:
  算子计算参数 `probs` 的张量描述，支持原位计算，支持多向广播。
- `target_desc` - { dT | (d1,...,dn) | (...) }:
  算子计算参数 `target` 的张量描述，支持原位计算，支持多向广播。

参数限制：

- `dT`:  (`Float16`, `Float32`, `Float64`, `BFloat16`) 之一。
- 输入 `probs` 与 `target` 的形状需与 `grad_logits` 相同。`probs` 与 `target` 涉及多向广播时需调整步长以匹配多向广播的映射关系。
- 支持原位计算，即计算时 `grad_logits` 可以和 `probs` 或 `target` 指向同一地址。
- 计算输出参数 `grad_logits` 不能进行广播（`grad_logits` 的步长不能涉及广播设置，即步长不能有 0）

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
  已使用 `infiniopCreateCrossEntropyLossBackwardDescriptor()` 初始化的算子描述符；
- `size`:
  额外空间大小的计算结果的写入地址；

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
  输入。 待销毁的算子描述符；

<div style="background-color: lightblue; padding: 1px;"> 返回值： </div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`].

## 已知问题

无

<!-- 链接 -->
[`InfiniopHandle_t`]: /infiniop/handle/README.md

[`INFINI_STATUS_SUCCESS`]: /common/status/README.md#INFINI_STATUS_SUCCESS
[`INFINI_STATUS_BAD_PARAM`]: /common/status/README.md#INFINI_STATUS_BAD_PARAM
[`INFINI_STATUS_INSUFFICIENT_WORKSPACE`]: /common/status/README.md#INFINI_STATUS_INSUFFICIENT_WORKSPACE
[`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`]: /common/status/README.md#INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED
[`INFINI_STATUS_INTERNAL_ERROR`]: /common/status/README.md#INFINI_STATUS_INTERNAL_ERROR
[`INFINI_STATUS_NULL_POINTER`]: /common/status/README.md#INFINI_STATUS_NULL_POINTER
[`INFINI_STATUS_BAD_TENSOR_SHAPE`]: /common/status/README.md#INFINI_STATUS_BAD_TENSOR_SHAPE
[`INFINI_STATUS_BAD_TENSOR_DTYPE`]: /common/status/README.md#INFINI_STATUS_BAD_TENSOR_DTYPE
[`INFINI_STATUS_BAD_TENSOR_STRIDES`]: /common/status/README.md#INFINI_STATUS_BAD_TENSOR_STRIDES
