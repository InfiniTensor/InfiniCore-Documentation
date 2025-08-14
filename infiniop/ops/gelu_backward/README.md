# `GELU Backward`

`GELU Backward`，即 **GELU 激活函数反向传播**算子，计算 GELU 函数的梯度。其计算可被表述为：

$$ \frac{\partial}{\partial x} \text{GELU}(x) = \Phi(x) + x \cdot \phi(x) $$

其中 $\Phi(x)$ 为标准正态分布的累积分布函数，$\phi(x)$ 为标准正态分布的概率密度函数。
采取近似计算

$$ \text{GELU}(x) = 0.5 \times x \times (1 + \text{Tanh}(\sqrt{\frac{2}{\pi}} \times (x + 0.044715 \times x^3))) $$
令
$$inner(x) =\sqrt{\frac{2}{\pi}} \times (x + 0.044715 \times x^3) $$
那么
$$ \frac{\partial}{\partial x} \text{GELU}(x)= 0.5 \times [ (1 + \text{Tanh}(inner)) + x \times ( 1- \text{Tanh}^2(inner)\times \frac{\partial }{\partial x}inner) ]$$

## 接口

### 计算

```c
infiniStatus_t infiniopGeluBackward(
    infiniopGeluBackwardDescriptor_t desc,
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
  已使用 `infiniopCreateGeluBackwardDescriptor()` 初始化的算子描述符；
- `workspace`:
  指向算子计算所需的额外工作空间；
- `workspace_size`:
  `workspace` 的大小，单位：字节；
- `grad_input`:
  输出梯度张量。张量限制见[创建算子描述](#创建算子描述)部分；
- `input`:
  前向传播的输入张量。张量限制见[创建算子描述](#创建算子描述)部分；
- `grad_output`:
  输入梯度张量。张量限制见[创建算子描述](#创建算子描述)部分；
- `stream`:
  计算流/队列；

<div style="background-color: lightblue; padding: 1px;"> 返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_BAD_PARAM`], [`INFINI_STATUS_INSUFFICIENT_WORKSPACE`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`], [`INFINI_STATUS_INTERNAL_ERROR`]，[`INFINI_STATUS_BAD_TENSOR_DTYPE`].

### 创建算子描述

```c
infiniStatus_t infiniopCreateGeluBackwardDescriptor(
    infiniopHandle_t handle,
    infiniopGeluBackwardDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t grad_input_desc,
    infiniopTensorDescriptor_t input_desc,
    infiniopTensorDescriptor_t grad_output_desc,
    bool approximate
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数：</div>

- `handle`:
  `infiniopHandle_t` 类型的硬件控柄。详情请看：[`InfiniopHandle_t`]。
- `desc_ptr`:
  `infiniopGeluBackwardDescriptor_t` 指针，指向将被初始化的算子描述符地址；
- `grad_input_desc` - { dT | (d1,...,dn) | (...) }:
  算子计算参数 `grad_input` 的张量描述，支持原位计算。
- `input_desc` - { dT | (d1,...,dn) | (...) }:
  算子计算参数 `input` 的张量描述。
- `grad_output_desc` - { dT | (d1,...,dn) | (...) }:
  算子计算参数 `grad_output` 的张量描述，支持原位计算。
- `approximate`:
  是否使用近似计算。`true` 表示使用 tanh 近似，`false` 表示使用精确的 erf 计算。

参数限制：

- `dT`:  (`Float16`, `Float32`, `Float64`, `BFloat16`) 之一。
- 所有张量的形状需相同。
- 支持原位计算，即计算时 `grad_input` 可以和 `grad_output` 指向同一地址。

<div style="background-color: lightblue; padding: 1px;"> 返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_BAD_PARAM`], [`INFINI_STATUS_BAD_TENSOR_SHAPE`], [`INFINI_STATUS_BAD_TENSOR_DTYPE`], [`INFINI_STATUS_BAD_TENSOR_STRIDES`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`].

### 计算额外工作空间

```c
infiniStatus_t infiniopGetGeluBackwardWorkspaceSize(
    infiniopGeluBackwardDescriptor_t desc,
    size_t *size
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数：</div>

- `desc`:
  已使用 `infiniopCreateGeluBackwardDescriptor()` 初始化的算子描述符；
- `size`:
  额外空间大小的计算结果的写入地址；

<div style="background-color: lightblue; padding: 1px;"> 返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_NULL_POINTER`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`].

### 销毁算子描述符

```c
infiniStatus_t infiniopDestroyGeluBackwardDescriptor(
    infiniopGeluBackwardDescriptor_t desc
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
