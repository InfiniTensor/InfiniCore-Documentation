# `Sigmoid Backward`

`Sigmoid Backward`，即 `Sigmoid` 激活函数的**反向传播算子**，用于计算输入 `input` 对应的梯度 `grad_input`，给定上游梯度 `grad_output`。其计算可被表述为：

令
$$
s = \sigma(\text{input}) = \frac{1}{1 + e^{-\text{input}}}
$$

则
$$
\text{grad\_input} = \text{grad\_output} \cdot s \cdot (1 - s)
$$

其中：
- `input`：前向传播的输入张量（用于计算 sigmoid 值 s ）  
- `grad_output`：上游传来的梯度（对 sigmoid 输出的梯度）  
- `grad_input`：本算子输出，即对 `input` 的梯度

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
- `grad_input：`:
  输出张量（对 `input` 的梯度）。张量限制见[创建算子描述](#创建算子描述)部分；
- `input`:
  输入张量。张量限制见[创建算子描述](#创建算子描述)部分；
- `grad_output：`:
  输入张量（上游梯度）。张量限制见[创建算子描述](#创建算子描述)部分；
- `stream`:
  计算流/队列；

<div style="background-color: lightblue; padding: 1px;"> 返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_BAD_PARAM`], [`INFINI_STATUS_INSUFFICIENT_WORKSPACE`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`], [`INFINI_STATUS_INTERNAL_ERROR`]，[`INFINI_STATUS_BAD_TENSOR_DTYPE`].

### 创建算子描述

```c
infiniopCreateSigmoidBackwardDescriptor(
    infiniopHandle_t handle,
    infiniopSigmoidBackwardDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t grad_input_desc,
    infiniopTensorDescriptor_t input_desc,
    infiniopTensorDescriptor_t grad_output_desc
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数：</div>

- `handle`:
  `infiniopHandle_t` 类型的硬件控柄。详情请看：[`InfiniopHandle_t`]。
- `desc_ptr`:
  `infiniopSigmoidBackwardDescriptor_t` 指针，指向将被初始化的算子描述符地址；
- `grad_input_desc` - { dT | (d1,...,dn) | (...) }:
  算子计算参数 `grad_input` 的张量描述，支持原位计算。
- `input_desc` - { dT | (d1,...,dn) | (...) }:
  算子计算参数 `input` 的张量描述，支持原位计算，支持多向广播。
- `grad_output_desc` - { dT | (d1,...,dn) | (...) }:
  算子计算参数 `grad_output` 的张量描述，支持原位计算，支持多向广播。  

参数限制：

- `dT`:  (`Float16`, `Float32`, `Float64`, `BFloat16`) 之一。
- 输入 `grad_output` 和输入 `input` 的形状需与 `grad_input` 相同。`grad_output` 和 `input` 涉及多向广播时需调整步长以匹配多向广播的映射关系。
- 支持原位计算，即计算时 `grad_input` 可以和 `grad_output` 、 `input` 指向同一地址。
- 计算输出参数 `grad_input` 不能进行广播（`grad_input` 的步长不能涉及广播设置，即步长不能有 0）

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
  已使用 `infiniopGetSigmoidBackwardWorkspaceSize()` 初始化的算子描述符；
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
