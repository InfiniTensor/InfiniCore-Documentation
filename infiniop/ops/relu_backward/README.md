
# `ReluBackward`

`ReluBackward`, 即**反向Relu**算子，为单目逐元素算子。其计算可被表述为：

```math
grad\_input = \begin{cases} 
    \text{grad\_output} & \text{if~ input} \gt 0 \\ 
    \text{0} & \text{if~ input} \leq 0  
\end{cases}
```

其中 `input` 和 `grad_output` 为输入，`grad_input` 为输出：

- `input`: 前向传播时Relu层的输入张量。
- `grad_output`：损失函数对Relu层输出的梯度。
- `grad_input`：损失函数对Relu层输入`input`的梯度。

## 接口

### 计算

```c
infiniStatus_t infiniopReluBackward(
    infiniopReluBackwardDescriptor_t desc,
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
  已使用 `infiniopCreateReluBackwardDescriptor()` 初始化的算子描述符；
- `workspace`:
  指向算子计算所需的额外工作空间；
- `workspace_size`:
  `workspace` 的大小，单位：字节；
- `grad_input`:
  输出张量。张量限制见[创建算子描述](#创建算子描述)部分；
- `input`:
  输入张量。张量限制见[创建算子描述](#创建算子描述)部分；
- `grad_output`:
  输入张量。张量限制见[创建算子描述](#创建算子描述)部分；
- `stream`:
  计算流/队列；

<div style="background-color: lightblue; padding: 1px;"> 返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_BAD_PARAM`], [`INFINI_STATUS_INSUFFICIENT_WORKSPACE`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`], [`INFINI_STATUS_INTERNAL_ERROR`]，[`INFINI_STATUS_BAD_TENSOR_DTYPE`].

### 创建算子描述

```c
infiniStatus_t infiniopCreateReluBackwardDescriptor(
    infiniopHandle_t handle,
    infiniopReluBackwardDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t grad_input_desc,
    infiniopTensorDescriptor_t input_desc,
    infiniopTensorDescriptor_t output_desc
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数：</div>

- `handle`:
  `infiniopHandle_t` 类型的硬件控柄。详情请看：[`InfiniopHandle_t`]。
- `desc_ptr`:
  `infiniopReluBackwardDescriptor_t` 指针，指向将被初始化的算子描述符地址；
- `grad_input_desc` - { dT | (d1,...,dn) | (...) }:
  算子计算参数 `grad_input` 的张量描述，支持原位计算。
- `input_desc` - { dT | (d1,...,dn) | (...) }:
  算子计算参数 `input` 的张量描述，支持原位计算，支持多向广播。
- `output_desc` - { dT | (d1,...,dn) | (...) }:
  算子计算参数 `grad_output` 的张量描述，支持原位计算，支持多向广播。

参数限制：

- `dT`:  (`BFloat16`,`Float16`, `Float32`, `Float64`) 之一。
- 输入 `input` 与 `grad_output` 的形状需与 `grad_input` 相同。`input` 与 `grad_output` 涉及多向广播时需调整步长以匹配多向广播的映射关系。
- 支持原位计算，即计算时 `grad_input` 可以和 `input` 或 `grad_output` 指向同一地址。
- 计算输出参数 `grad_input` 不能进行广播（`grad_input` 的步长不能涉及广播设置，即步长不能有 0）

<div style="background-color: lightblue; padding: 1px;"> 返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_BAD_PARAM`], [`INFINI_STATUS_BAD_TENSOR_SHAPE`], [`INFINI_STATUS_BAD_TENSOR_DTYPE`], [`INFINI_STATUS_BAD_TENSOR_STRIDES`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`].

### 计算额外工作空间

```c
infiniStatus_t infiniopGetReluBackwardWorkspaceSize(
    infiniopReluBackwardDescriptor_t desc,
    size_t *size
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数：</div>

- `desc`:
  已使用 `infiniopCreateReluBackwardDescriptor()` 初始化的算子描述符；
- `size`:
  额外空间大小的计算结果的写入地址；

<div style="background-color: lightblue; padding: 1px;"> 返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_NULL_POINTER`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`].

### 销毁算子描述符

```c
infiniStatus_t infiniopDestroyReluBackwardDescriptor(
    infiniopReluBackwardDescriptor_t desc
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
