
# `RMS Norm Backward`

`RMS Norm Backward`, 是`RMS Norm`算子所对应的反向传播算子。其正向公式如下：

$$
  y_i=\frac{x_i}{\sqrt{\frac{1}{\rm{D}}\sum_{i=1}^{\rm{D}}x_i^2}}\cdot \gamma_i
$$

其中分母是针对3D输入张量的第三维向量的二范数，D为输入张量第三维的长度。

## 接口

### 计算

```c
infiniStatus_t infiniopRMSNormBackward(
    infiniopRMSNormBackwardDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void * grad_x,
    void * grad_w,
    const void * grad_y,
    const void * x,
    const void * w,
    void *stream
);
```
<div style="background-color: lightblue; padding: 1px;"> 参数： </div>

- `desc`:
  已使用 `infiniopCreateRMSNormBackwardDescriptor()` 初始化的算子描述符。
- `workspace`:
  指向算子计算所需的额外工作空间；
- `workspace_size`:
  `workspace` 的大小，单位：字节；
- `grad_x`:输出张量。张量限制见[创建算子描述](#创建算子描述)部分。
- `grad_w`:输出张量。张量限制见[创建算子描述](#创建算子描述)部分。
- `grad_y`:输入张量。张量限制见[创建算子描述](#创建算子描述)部分。
- `x`:输入张量。张量限制见[创建算子描述](#创建算子描述)部分。
- `w`:输入张量。张量限制见[创建算子描述](#创建算子描述)部分。
 - `stream`: 计算流/队列。

<div style="background-color: lightblue; padding: 1px;">  返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_BAD_PARAM`], [`INFINI_STATUS_INSUFFICIENT_WORKSPACE`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`], [`INFINI_STATUS_INTERNAL_ERROR`]，[`INFINI_STATUS_BAD_TENSOR_DTYPE`].

---

### 创建算子描述

```c
infiniStatus_t infiniopCreateRMSNormBackwardDescriptor(
    infiniopHandle_t handle,
    infiniopRMSNormBackwardDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t grad_x_desc,
    infiniopTensorDescriptor_t grad_w_desc,
    infiniopTensorDescriptor_t grad_y_desc,
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t w_desc
);
```
<div style="background-color: lightblue; padding: 1px;"> 参数：</div>

- `handle`:
  `infiniopHandle_t` 类型的硬件控柄。详情请看：[`InfiniopHandle_t`]。
- `desc_ptr`:
  `infiniopAddDescriptor_t` 指针，指向将被初始化的算子描述符地址；
- `grad_x_desc` - { dT | (N1, N2, D) | (...,1) }:
     算子计算参数 `grad_x` 的张量描述，对应于`x`的梯度。
- `grad_w_desc` - { dT | (D) | (...) }:
     算子计算参数 `grad_w` 的张量描述，对应于公式中 $\gamma$ 的梯度。
- `grad_y_desc` - { dT | (N1, N2, D) | (...,1) }:
     算子计算参数 `grad_y` 的张量描述，对应于公式中`y`的梯度。
- `x_desc` - { dT | (N1, N2, D) | (...,1) }:
     算子计算参数 `x` 的张量描述。
- `w_desc` - { dT | (D) | (...) }:
     算子计算参数 `w` 的张量描述，对应于公式中的 $\gamma$。

参数限制：

- **`dT`**:  (`Float16`, `Float32`, `Bfloat16`) 之一
- `grad_x`、`grad_y`、`x`的最后一维均连续

<div style="background-color: lightblue; padding: 1px;"> 返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_BAD_PARAM`], [`INFINI_STATUS_BAD_TENSOR_SHAPE`], [`INFINI_STATUS_BAD_TENSOR_DTYPE`], [`INFINI_STATUS_BAD_TENSOR_STRIDES`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`].
---

### 计算额外工作空间

```c
infiniStatus_t infiniopGetRMSNormBackwardWorkspaceSize(
    infiniopRMSNormBackwardDescriptor_t desc,
    size_t *size
);
```
<div style="background-color: lightblue; padding: 1px;"> 参数：</div>

- `desc`: 使用 `infiniopCreateRMSNormBackwardDescriptor()` 初始化的算子描述符。
- `size`:
  额外空间大小的计算结果的写入地址；

<div style="background-color: lightblue; padding: 1px;"> 返回值：</div>

 - [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_BAD_PARAM`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`].

---

### 销毁算子描述符

```c
infiniopStatus_t infiniopDestroyRMSNormBackwardDescriptor(
    infiniopRMSNormBackwardDescriptor_t desc
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
