
# `Batch Norm`

`Batch Norm`, 对3D输入张量在中间维度进行归一化和移动放缩。其公式如下:

$$
     y=\frac{x-{\rm{E}}\left[x\right]}{\sqrt{{\rm{Var}}\left[x\right]+\epsilon}}\cdot\gamma+\beta
$$

其中`x`为输入元素，$\epsilon$ 是一个小的常数，用于避免除以零。$\gamma$ 和 $\beta$ 为权重张量和平移张量。

`E[x]`和`Var[x]`为针对第一、第三维度的均值和方差，其更新根据规则如下：
$$
     {\rm{E}}\left[x\right]_{\rm{new}} = (1 - {\rm{momentum}})\cdot{\rm{E}}\left[x\right]_{\rm{old}} + {\rm{momentum}} \cdot{\rm{E}}\left[x\right]
$$
$$
     {\rm{Var}}\left[x\right]_{\rm{new}} = (1 - {\rm{momentum}})\cdot{\rm{Var}}\left[x\right]_{\rm{old}} + {\rm{momentum}} \cdot{\rm{Var}}\left[x\right]
$$

其中`momentum`为设定动量参数。

## 接口

### 计算

```c
infiniStatus_t infiniopBatchNorm(
    infiniopBatchNormDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void * output,
    void * running_mean,
    void * running_var,
    const void * input,
    const void * weight,
    const void * bias,
    void *stream
);
```
<div style="background-color: lightblue; padding: 1px;"> 参数： </div>

- `desc`:
  已使用 `infiniopCreateBatchNormDescriptor()` 初始化的算子描述符。
- `workspace`:
  指向算子计算所需的额外工作空间；
- `workspace_size`:
  `workspace` 的大小，单位：字节；
- `output`:输出张量。张量限制见[创建算子描述](#创建算子描述)部分。
- `running_mean`:输出张量。张量限制见[创建算子描述](#创建算子描述)部分。
- `running_var`:输出张量。张量限制见[创建算子描述](#创建算子描述)部分。
- `input`:输入张量。张量限制见[创建算子描述](#创建算子描述)部分。
- `weight`:输入张量。张量限制见[创建算子描述](#创建算子描述)部分。
- `bias`:输入张量。张量限制见[创建算子描述](#创建算子描述)部分。
 - `stream`: 计算流/队列。

<div style="background-color: lightblue; padding: 1px;">  返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_BAD_PARAM`], [`INFINI_STATUS_INSUFFICIENT_WORKSPACE`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`], [`INFINI_STATUS_INTERNAL_ERROR`]，[`INFINI_STATUS_BAD_TENSOR_DTYPE`].

---

### 创建算子描述

```c
infiniStatus_t infiniopCreateBatchNormDescriptor(
    infiniopHandle_t handle,
    infiniopBatchNormDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t running_mean_desc,
    infiniopTensorDescriptor_t running_var_desc,
    infiniopTensorDescriptor_t input_desc,
    infiniopTensorDescriptor_t weight_desc,
    infiniopTensorDescriptor_t bias_desc,
    float momentum,
    float eps
);
```
<div style="background-color: lightblue; padding: 1px;"> 参数：</div>

- `handle`:
  `infiniopHandle_t` 类型的硬件控柄。详情请看：[`InfiniopHandle_t`]。
- `desc_ptr`:
  `infiniopAddDescriptor_t` 指针，指向将被初始化的算子描述符地址；
- `output_desc` - { dT | (N, C, L) | (C $\cdot$ L, L, 1) }:
     算子计算参数 `output` 的张量描述， N是批次大小， C是特征或通道数， L是序列长度。
- `running_mean_desc` - { dT | (C) | (...) }:
     算子计算参数 `running_mean` 的张量描述，对应公式中的 ${\rm{E}}\left[x\right]$。
- `running_var_desc` - { dT | (C) | (...) }:
     算子计算参数 `running_var` 的张量描述，对应公式中的 ${\rm{Var}}\left[x\right]$。
- `input_desc` - { dT | (N, C, L) | (C $\cdot$ L, L, 1) }:
     算子计算参数 `input` 的张量描述。
- `weight_desc` - { dT | (C) | (...) }:
     算子计算参数 `weight` 的张量描述，对应公式中的 $\gamma$。
- `bias_desc` - { dT | (C) | (...) }:
     算子计算参数 `bias` 的张量描述，对应公式中的 $\beta$。
- `momentum` - float: 即公式中的 `momentum`。
- `eps` - float: 即公式中的 $\epsilon$。

参数限制：

- **`dT`**:  (`Float16`, `Float32`, `Bfloat16`) 之一
- `output`与`input`均为全连续张量。

<div style="background-color: lightblue; padding: 1px;"> 返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_BAD_PARAM`], [`INFINI_STATUS_BAD_TENSOR_SHAPE`], [`INFINI_STATUS_BAD_TENSOR_DTYPE`], [`INFINI_STATUS_BAD_TENSOR_STRIDES`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`].
---

### 计算额外工作空间

```c
infiniStatus_t infiniopGetBatchNormWorkspaceSize(
    infiniopBatchNormDescriptor_t desc,
    size_t *size
);
```
<div style="background-color: lightblue; padding: 1px;"> 参数：</div>

- `desc`: 使用 `infiniopCreateBatchNormDescriptor()` 初始化的算子描述符。
- `size`:
  额外空间大小的计算结果的写入地址；

<div style="background-color: lightblue; padding: 1px;"> 返回值：</div>

 - [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_BAD_PARAM`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`].

---

### 销毁算子描述符

```c
infiniopStatus_t infiniopDestroyBatchNormDescriptor(
    infiniopBatchNormDescriptor_t desc
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
