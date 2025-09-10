# Softplus 算子

## 数学定义

给定输入张量 $x$，Softplus 算子的计算公式为：

$$\text{softplus}(x) = \log(1 + e^x)$$

为了保证数值稳定性，在 $x>20$ 时，算子返回 $x$ （即线性计算）。

## 接口

### 计算

```c
infiniStatus_t infiniopSoftplus(
    infiniopSoftplusDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *input,
    void *stream
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数：</div>

- `desc`:
  已使用 `infiniopCreateSoftplusDescriptor()` 初始化的算子描述符。
- `workspace`:
  额外工作空间地址。
- `workspace_size`:
  `workspace` 的大小，单位：字节。
- `output`:
  计算结果地址，当与 `input` 相同时为原地操作。
- `input`:
  输入张量的数据指针。
- `stream`:
  计算流/队列。

<div style="background-color: lightblue; padding: 1px;"> 返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_NULL_POINTER`], [`INFINI_STATUS_INSUFFICIENT_WORKSPACE`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`], [`INFINI_STATUS_INTERNAL_ERROR`].

### 创建算子描述

```c
infiniStatus_t infiniopCreateSoftplusDescriptor(
    infiniopHandle_t handle,
    infiniopSoftplusDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t output,
    infiniopTensorDescriptor_t input
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数：</div>

- `handle`:
  硬件控柄。详见 [`InfiniopHandle_t`]。
- `desc_ptr`:
  `infiniopSoftplusDescriptor_t` 指针，指向将被初始化的算子描述符地址。
- `output` - { dT | (d1,...,dn) | (...) }:
  计算结果 `output` 的张量描述，支持原位计算。
- `input` - { dT | (d1,...,dn) | (...) }:
  输入张量 `input` 的张量描述。

参数限制：

- `dT`: (`Float16`, `Float32`, `BFloat16`) 之一。
- 输入和输出张量必须具有相同的形状和数据类型。
- 支持原位计算，即计算时 `output` 可以和 `input` 指向同一地址。

<div style="background-color: lightblue; padding: 1px;"> 返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_BAD_PARAM`], [`INFINI_STATUS_BAD_TENSOR_SHAPE`], [`INFINI_STATUS_BAD_TENSOR_DTYPE`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`].

### 计算额外工作空间

```c
infiniStatus_t infiniopGetSoftplusWorkspaceSize(
    infiniopSoftplusDescriptor_t desc,
    size_t *size
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数：</div>

- `desc`:
  已使用 `infiniopCreateSoftplusDescriptor()` 初始化的算子描述符。
- `size`:
  额外空间大小的计算结果的写入地址。

<div style="background-color: lightblue; padding: 1px;"> 返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_NULL_POINTER`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`].

### 销毁算子描述符

```c
infiniStatus_t infiniopDestroySoftplusDescriptor(
    infiniopSoftplusDescriptor_t desc
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
[`INFINI_STATUS_NULL_POINTER`]:/common/status/README.md#INFINI_STATUS_NULL_POINTER
[`INFINI_STATUS_INSUFFICIENT_WORKSPACE`]:/common/status/README.md#INFINI_STATUS_INSUFFICIENT_WORKSPACE
[`INFINI_STATUS_INTERNAL_ERROR`]:/common/status/README.md#INFINI_STATUS_INTERNAL_ERROR
