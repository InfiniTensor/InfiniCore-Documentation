# `CrossEntropyLoss Backward`

`CrossEntropyLoss Backward`，即 **交叉熵损失*- 的反向算子。梯度按元素计算如下（假设 `probs` 为已归一化的类别概率，`target` 为与 `logits` 同形状的 one-hot 张量）：

$$
\textbf{grad\_logits}=\frac{\textbf{probs}-\textbf{target}}{N},
$$

其中 $N$ 为归一化因子（常见取法：对批量求平均时 $N=$ 批大小；逐样本/逐位置计算时可取 $N=1$）。

`probs` 与 `target` 为输入，`grad_logits` 为输出。

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

<div style="background-color: lightblue; padding: 1px;">参数：</div>

- `desc`:
  已使用 `infiniopCreateCrossEntropyLossBackwardDescriptor()` 初始化的算子描述符；
- `workspace`:
  指向算子计算所需的额外工作空间；
- `workspace_size`:
  `workspace` 的大小，单位：字节；
- `grad_logits`:
  输出张量（∂L/∂logits）。张量限制见[创建算子描述](#创建算子描述)部分；
- `probs`:
  概率张量（通常为 softmax 输出）。张量限制见[创建算子描述](#创建算子描述)部分；
- `target`:
  one-hot 目标张量（形状与 `logits` 相同）。张量限制见[创建算子描述](#创建算子描述)部分；
- `stream`:
  计算流/队列；

<div style="background-color: lightblue; padding: 1px;">返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_BAD_PARAM`], [`INFINI_STATUS_INSUFFICIENT_WORKSPACE`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`], [`INFINI_STATUS_INTERNAL_ERROR`], [`INFINI_STATUS_BAD_TENSOR_DTYPE`].

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

<div style="background-color: lightblue; padding: 1px;">参数：</div>

- `handle`:
  `infiniopHandle_t` 类型的硬件控柄。详情请看：[`InfiniopHandle_t`]。
- `desc_ptr`:
  `infiniopCrossEntropyLossBackwardDescriptor_t` 指针，指向将被初始化的算子描述符地址；
- `grad_logits_desc` - { dT | (d1,...,dn) | (...) }:
  输出 `grad_logits` 的张量描述，支持原位计算；
- `probs_desc` - { dT | (d1,...,dn) | (...) }:
  输入 `probs` 的张量描述，支持原位计算；
- `target_desc` - { dT | (d1,...,dn) | (...) }:
  输入 `target` 的张量描述，支持原位计算；

参数限制：

- `dT`: (`Float16`, `Float32`, `Float64`, `BFloat16`) 之一。
- 输入 `probs` 与 `target` 的形状需与 `grad_logits` 相同。
- 支持原位计算,即计算时`grad_logits` 可与 `probs` 或 `target` 指向同一地址。

<div style="background-color: lightblue; padding: 1px;">返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_BAD_PARAM`], [`INFINI_STATUS_BAD_TENSOR_SHAPE`], [`INFINI_STATUS_BAD_TENSOR_DTYPE`], [`INFINI_STATUS_BAD_TENSOR_STRIDES`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`].

### 计算额外工作空间

```c
infiniStatus_t infiniopGetCrossEntropyLossBackwardWorkspaceSize(
    infiniopCrossEntropyLossBackwardDescriptor_t desc,
    size_t *size
);
```

<div style="background-color: lightblue; padding: 1px;">参数：</div>

- `desc`:
  已使用 `infiniopCreateCrossEntropyLossBackwardDescriptor()` 初始化的算子描述符；
- `size`:
  额外空间大小的计算结果的写入地址；

<div style="background-color: lightblue; padding: 1px;">返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_NULL_POINTER`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`].

### 销毁算子描述符

```c
infiniStatus_t infiniopDestroyCrossEntropyLossBackwardDescriptor(
    infiniopCrossEntropyLossBackwardDescriptor_t desc
);
```

<div style="background-color: lightblue; padding: 1px;">参数：</div>

- `desc`:
  输入。待销毁的算子描述符；

<div style="background-color: lightblue; padding: 1px;">返回值：</div>

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
