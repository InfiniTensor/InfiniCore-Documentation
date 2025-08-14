# `CrossEntropyLoss Backward`

`CrossEntropyLoss Backward`，即 **交叉熵损失函数反向传播**算子，计算交叉熵损失函数的梯度。
计算为：
$$grad\_logits = (probs - target) / N$$ 
其中probs为概率，target为形状与logits相同的one-hot张量
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
  输出梯度张量（对 logits 的梯度）。张量限制见[创建算子描述](#创建算子描述)部分；
- `probs`:
  输入概率张量（softmax 后的概率分布）。张量限制见[创建算子描述](#创建算子描述)部分；
- `target`:
  目标标签张量。张量限制见[创建算子描述](#创建算子描述)部分；
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
    infiniopTensorDescriptor_t target_desc,
    int64_t reduction,
    int64_t ignore_index
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数：</div>

- `handle`:
  `infiniopHandle_t` 类型的硬件控柄。详情请看：[`InfiniopHandle_t`]。
- `desc_ptr`:
  `infiniopCrossEntropyLossBackwardDescriptor_t` 指针，指向将被初始化的算子描述符地址；
- `grad_logits_desc` - { dT | (N, C) | (...) }:
  算子计算参数 `grad_logits` 的张量描述。
- `probs_desc` - { dT | (N, C) | (...) }:
  算子计算参数 `probs` 的张量描述。
- `target_desc` - { iT | (N,) | (...) }:
  算子计算参数 `target` 的张量描述。
- `reduction`:
  损失缩减方式。0: 无缩减，1: 求平均，2: 求和。
- `ignore_index`:
  忽略的目标索引值，通常用于忽略填充标记。

参数限制：

- `dT`:  (`Float16`, `Float32`, `Float64`, `BFloat16`) 之一。
- `iT`:  (`Int32`, `Int64`) 之一。
- `probs` 张量形状为 (N, C)，其中 N 为批次大小，C 为类别数。
- `target` 张量形状为 (N,)，包含类别索引。
- `grad_logits` 张量形状为 (N, C)，与 `probs` 形状相同。

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
