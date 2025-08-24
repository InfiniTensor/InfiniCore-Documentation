# `Sigmoid Backward`

`Sigmoid Backward`, 即**Sigmoid反向传播**算子，用于计算Sigmoid函数的梯度。其计算可被表述为：

$$ grad\_input = grad\_output \times sigmoid(input) \times (1 - sigmoid(input)) $$

其中 `input` 为前向传播的输入，`grad_output` 为从后续层传回的梯度，`grad_input` 为计算得到的输入梯度。

## 接口

### 计算

```c
infiniStatus_t infiniopSigmoidBackward(
    infiniopSigmoidBackwardDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *grad_input,
    const void *input,
    const void *grad_output,
    void *stream
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数： </div>

- `desc`:
  已使用 `infiniopCreateSigmoidBackwardDescriptor()` 初始化的算子描述符；
- `workspace`:
  指向算子计算所需的额外工作空间；
- `workspace_size`:
  `workspace` 的大小，单位：字节；
- `grad_input`:
  输出的输入梯度张量。张量限制见[创建算子描述](#创建算子描述)部分；
- `input`:
  前向传播的输入张量。张量限制见[创建算子描述](#创建算子描述)部分；
- `grad_output`:
  从后续层传回的梯度张量。张量限制见[创建算子描述](#创建算子描述)部分；
- `stream`:
  计算流/队列；

<div style="background-color: lightblue; padding: 1px;"> 返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_BAD_PARAM`], [`INFINI_STATUS_INSUFFICIENT_WORKSPACE`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`], [`INFINI_STATUS_INTERNAL_ERROR`]，[`INFINI_STATUS_BAD_TENSOR_DTYPE`].

### 创建算子描述

```c
infiniStatus_t infiniopCreateSigmoidBackwardDescriptor(
    infiniopHandle_t handle,
    infiniopSigmoidBackwardDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t grad_input,
    infiniopTensorDescriptor_t input,
    infiniopTensorDescriptor_t grad_output
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数：</div>

- `handle`:
  `infiniopHandle_t` 类型的硬件控柄。详情请看：[`InfiniopHandle_t`]。
- `desc_ptr`:
  `infiniopSigmoidBackwardDescriptor_t` 指针，指向将被初始化的算子描述符地址；
- `grad_input` - { dT | (d1,...,dn) | (...) }:
  算子计算参数 `grad_input` 的张量描述，支持原位计算。
- `input` - { dT | (d1,...,dn) | (...) }:
  算子计算参数 `input` 的张量描述。
- `grad_output` - { dT | (d1,...,dn) | (...) }:
  算子计算参数 `grad_output` 的张量描述，支持原位计算。

参数限制：

- `dT`:  (`Float16`, `Float32`, `BFloat16`) 之一。
- 所有张量 `input`、`grad_output` 与 `grad_input` 的形状需相同。
- 支持原位计算，即计算时 `grad_input` 可以和 `grad_output` 指向同一地址。
- 所有张量类型一致。

<div style="background-color: lightblue; padding: 1px;"> 返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_BAD_PARAM`], [`INFINI_STATUS_BAD_TENSOR_SHAPE`], [`INFINI_STATUS_BAD_TENSOR_DTYPE`], [`INFINI_STATUS_BAD_TENSOR_STRIDES`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`].

### 计算额外工作空间

```c
infiniStatus_t infiniopGetSigmoidBackwardWorkspaceSize(
    infiniopSigmoidBackwardDescriptor_t desc,
    size_t *size
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数：</div>

- `desc`:
  已使用 `infiniopCreateSigmoidBackwardDescriptor()` 初始化的算子描述符；
- `size`:
  额外空间大小的计算结果的写入地址；

<div style="background-color: lightblue; padding: 1px;"> 返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_NULL_POINTER`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`].

### 销毁算子描述符

```c
infiniStatus_t infiniopDestroySigmoidBackwardDescriptor(
    infiniopSigmoidBackwardDescriptor_t desc
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