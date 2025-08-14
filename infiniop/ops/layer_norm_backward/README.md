
# `Layer Norm Backward`

`Layer Norm Backward`, 是`Layer Norm`算子对应的反向传播算子。其正向算子的计算公式如下:

$$
     y=\frac{x-{\rm{E}}\left[x\right]}{\sqrt{{\rm{Var}}\left[x\right]+\epsilon}}\cdot\gamma+\beta
$$

其中`x`为输入元素，$\epsilon$ 是一个小的常数，用于避免除以零。$\gamma$ 和 $\beta$ 为权重张量和平移张量。
`E[x]`和`Var[x]`为针对最后一维的均值和方差。

## 接口

### 计算

```c
infiniStatus_t infiniopLayerNormBackward(
    infiniopLayerNormBackwardDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void * grad_input,
    void * grad_weight,
    void * grad_bias,
    const void * grad_output,
    const void * weight,
    const void * input_standardization,
    const void * input_std_deviation,
    void *stream
);
```
<div style="background-color: lightblue; padding: 1px;"> 参数： </div>

- `desc`:
  已使用 `infiniopCreateLayerNormBackwardDescriptor()` 初始化的算子描述符。
- `workspace`:
  指向算子计算所需的额外工作空间；
- `workspace_size`:
  `workspace` 的大小，单位：字节；
- `grad_input`:输出张量。张量限制见[创建算子描述](#创建算子描述)部分。
- `grad_weight`:输出张量。张量限制见[创建算子描述](#创建算子描述)部分。
- `grad_bias`:输出张量。张量限制见[创建算子描述](#创建算子描述)部分。
- `grad_output`:输入张量。张量限制见[创建算子描述](#创建算子描述)部分。
- `weight`:输入张量。张量限制见[创建算子描述](#创建算子描述)部分。
- `input_standardization`:输入张量。张量限制见[创建算子描述](#创建算子描述)部分。
- `input_std_deviation`:输入张量。张量限制见[创建算子描述](#创建算子描述)部分。
 - `stream`: 计算流/队列。

<div style="background-color: lightblue; padding: 1px;">  返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_BAD_PARAM`], [`INFINI_STATUS_INSUFFICIENT_WORKSPACE`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`], [`INFINI_STATUS_INTERNAL_ERROR`]，[`INFINI_STATUS_BAD_TENSOR_DTYPE`].

---

### 创建算子描述

```c
infiniStatus_t infiniopCreateLayerNormBackwardDescriptor(
    infiniopHandle_t handle,
    infiniopLayerNormBackwardDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t grad_input_desc,
    infiniopTensorDescriptor_t grad_weight_desc,
    infiniopTensorDescriptor_t grad_bias_desc,
    infiniopTensorDescriptor_t grad_output_desc,
    infiniopTensorDescriptor_t weight_desc,
    infiniopTensorDescriptor_t input_standardization_desc,
    infiniopTensorDescriptor_t input_std_deviation_desc
);
```
<div style="background-color: lightblue; padding: 1px;"> 参数：</div>

- `handle`:
  `infiniopHandle_t` 类型的硬件控柄。详情请看：[`InfiniopHandle_t`]。
- `desc_ptr`:
  `infiniopAddDescriptor_t` 指针，指向将被初始化的算子描述符地址；
- `grad_input_desc` - { dT | (N1, N2, L) | (...,1) }:
     算子计算参数 `grad_input` 的张量描述， 。
- `grad_weight_desc` - { dT | (L) | (...) }:
     算子计算参数 `grad_weight` 的张量描述， 。
- `grad_bias_desc` - { dT | (L) | (...) }:
     算子计算参数 `grad_bias` 的张量描述， 。
- `grad_output_desc` - { dT | (N1, N2, L) | (...,1) }:
     算子计算参数 `grad_output` 的张量描述， 。
- `weight_desc` - { dT | (L) | (...) }:
     算子计算参数 `weight` 的张量描述， 。
- `input_standardization_desc` - { dT | (N1, N2, L) | (..., 1) }:
     算子计算参数 `input_standardization` 的张量描述，，对应公式中的`input`归一化的结果: $\left({x-{\rm{E}}\left[x\right]}\right)/{\sqrt{{\rm{Var}}\left[x\right]+\epsilon}}$。
- `input_std_deviation_desc` - { dT | (N1, N2) | (...) }:
     算子计算参数 `input_std_deviation` 的张量描述，对应公式中的标准差: $\sqrt{{\rm{Var}}\left[x\right]+\epsilon}$。

参数限制：

- **`dT`**:  (`Float16`, `Float32`, `Bfloat16`) 之一
- `grad_input_desc`、`grad_output_desc`和 `input_standardization_desc`的最后一维要求连续。


<div style="background-color: lightblue; padding: 1px;"> 返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_BAD_PARAM`], [`INFINI_STATUS_BAD_TENSOR_SHAPE`], [`INFINI_STATUS_BAD_TENSOR_DTYPE`], [`INFINI_STATUS_BAD_TENSOR_STRIDES`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`].
---

### 计算额外工作空间

```c
infiniStatus_t infiniopGetLayerNormBackwardWorkspaceSize(
    infiniopLayerNormBackwardDescriptor_t desc,
    size_t *size
);
```
<div style="background-color: lightblue; padding: 1px;"> 参数：</div>

- `desc`: 使用 `infiniopCreateLayerNormBackwardDescriptor()` 初始化的算子描述符。
- `size`:
  额外空间大小的计算结果的写入地址；

<div style="background-color: lightblue; padding: 1px;"> 返回值：</div>

 - [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_BAD_PARAM`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`].

---

### 销毁算子描述符

```c
infiniopStatus_t infiniopDestroyLayerNormBackwardDescriptor(
    infiniopLayerNormBackwardDescriptor_t desc
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
