
# `Batch Norm Backward`

`Batch Norm Backward`, 即`Batch Norm`算子所对应的反向传播算子。其正向公式如下：
$$
     y=\frac{x-{\rm{E}}\left[x\right]}{\sqrt{{\rm{Var}}\left[x\right]}}\cdot\gamma+\beta
$$

其中`x`为输入元素。$\gamma$ 和 $\beta$ 为权重张量和平移张量。

`E[x]`和`Var[x]`为同时针对第一、第三维度的均值和方差。

## 接口

### 计算

```c
infiniStatus_t infiniopBatchNormBackward(
    infiniopBatchNormBackwardDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void * grad_input,
    void * grad_weight,
    void * grad_bias,
    const void * input,
    const void * grad_output,
    const void * weight,
    const void * running_mean,
    const void * running_var,
    void *stream
);
```
<div style="background-color: lightblue; padding: 1px;"> 参数： </div>

- `desc`:
  已使用 `infiniopCreateBatchNormBackwardDescriptor()` 初始化的算子描述符。
- `workspace`:
  指向算子计算所需的额外工作空间；
- `workspace_size`:
  `workspace` 的大小，单位：字节；
- `grad_input`:输出张量。张量限制见[创建算子描述](#创建算子描述)部分。
- `grad_weight`:输出张量。张量限制见[创建算子描述](#创建算子描述)部分。
- `grad_bias`:输出张量。张量限制见[创建算子描述](#创建算子描述)部分。
- `input`:输入张量。张量限制见[创建算子描述](#创建算子描述)部分。
- `grad_output`:输入张量。张量限制见[创建算子描述](#创建算子描述)部分。
- `weight`:输入张量。张量限制见[创建算子描述](#创建算子描述)部分。
- `running_mean`:输入张量。张量限制见[创建算子描述](#创建算子描述)部分。
- `running_var`:输入张量。张量限制见[创建算子描述](#创建算子描述)部分。
 - `stream`: 计算流/队列。

<div style="background-color: lightblue; padding: 1px;">  返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_BAD_PARAM`], [`INFINI_STATUS_INSUFFICIENT_WORKSPACE`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`], [`INFINI_STATUS_INTERNAL_ERROR`]，[`INFINI_STATUS_BAD_TENSOR_DTYPE`].

---

### 创建算子描述

```c
infiniStatus_t infiniopCreateBatchNormBackwardDescriptor(
    infiniopHandle_t handle,
    infiniopBatchNormBackwardDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t grad_input_desc,
    infiniopTensorDescriptor_t grad_weight_desc,
    infiniopTensorDescriptor_t grad_bias_desc,
    infiniopTensorDescriptor_t input_desc,
    infiniopTensorDescriptor_t grad_output_desc,
    infiniopTensorDescriptor_t weight_desc,
    infiniopTensorDescriptor_t running_mean_desc,
    infiniopTensorDescriptor_t running_var_desc
);
```
<div style="background-color: lightblue; padding: 1px;"> 参数：</div>

- `handle`:
  `infiniopHandle_t` 类型的硬件控柄。详情请看：[`InfiniopHandle_t`]。
- `desc_ptr`:
  `infiniopAddDescriptor_t` 指针，指向将被初始化的算子描述符地址；
- `grad_input_desc` - { dT | (N, C, L) | (C $\cdot$ L, L, 1) }:
     算子计算参数 `output` 的张量描述，对应输入张量`input`的梯度，N是批次大小， C是特征或通道数， L是序列长度。
- `grad_weight_desc` - { dT | (C) | (...) }:
     算子计算参数 `grad_weight` 的张量描述, 对应公式中权重向量 $\gamma$ 的梯度。
- `grad_bias_desc` - { dT | (C) | (...) }:
     算子计算参数 `grad_bias` 的张量描述, 对应公式中向量 $\beta$ 的梯度。
- `input_desc` - { dT | (N, C, L) | (C $\cdot$ L, L, 1) }:
     算子计算参数 `input` 的张量描述。
- `grad_output_desc` - { dT | (N, C, L) | (C $\cdot$ L, L, 1) }:
     算子计算参数 `grad_output` 的张量描述, 对应`output`的梯度。
- `weight_desc` - { dT | (C) | (...) }:
     算子计算参数 `weight` 的张量描述，对应公式中权重向量 $\gamma$ 。
- `running_mean_desc` - { dT | (C) | (...) }:
     算子计算参数 `running_mean` 的张量描述， 对应于${\rm{E}}[x]$,即`input`针对中间维度的均值。
- `running_var_desc` - { dT | (C) | (...) }:
     算子计算参数 `running_var` 的张量描述， 对应于${\rm{Var}}[x]$,即`input`针对中间维度的方差。

参数限制：

- **`dT`**:  (`Float16`, `Float32`, `Bfloat16`) 之一
- `grad_input`、`grad_output`、`input`均为全连续张量。

<div style="background-color: lightblue; padding: 1px;"> 返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_BAD_PARAM`], [`INFINI_STATUS_BAD_TENSOR_SHAPE`], [`INFINI_STATUS_BAD_TENSOR_DTYPE`], [`INFINI_STATUS_BAD_TENSOR_STRIDES`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`].
---

### 计算额外工作空间

```c
infiniStatus_t infiniopGetBatchNormBackwardWorkspaceSize(
    infiniopBatchNormBackwardDescriptor_t desc,
    size_t *size
);
```
<div style="background-color: lightblue; padding: 1px;"> 参数：</div>

- `desc`: 使用 `infiniopCreateBatchNormBackwardDescriptor()` 初始化的算子描述符。
- `size`:
  额外空间大小的计算结果的写入地址；

<div style="background-color: lightblue; padding: 1px;"> 返回值：</div>

 - [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_BAD_PARAM`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`].

---

### 销毁算子描述符

```c
infiniopStatus_t infiniopDestroyBatchNormBackwardDescriptor(
    infiniopBatchNormBackwardDescriptor_t desc
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
