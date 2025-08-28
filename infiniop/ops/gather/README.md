
# `Gather`

对于输入张量`input`、`index`和输出张量`output`，`Gather` 算子为`output`的每个位置赋值，所赋之值在`input`中的来源位置由 `index` 指定。对于 `output` 中的每个值，其来源位置由 `output` 中相对于 dimension != `dim` 的索引，以及 `index` 中对应的值指定。

## 接口

### 计算

```c
infiniStatus_t infiniopGather(
    infiniopGatherDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void * output,
    const void * input,
    const void * index,
    void *stream
);
```
<div style="background-color: lightblue; padding: 1px;"> 参数： </div>

- `desc`:
  已使用 `infiniopCreateGatherDescriptor()` 初始化的算子描述符。
- `workspace`:
  指向算子计算所需的额外工作空间；
- `workspace_size`:
  `workspace` 的大小，单位：字节；
- `output`:输出张量。张量限制见[创建算子描述](#创建算子描述)部分。
- `input`:输入张量。张量限制见[创建算子描述](#创建算子描述)部分。
- `index`:输入张量。张量限制见[创建算子描述](#创建算子描述)部分。
 - `stream`: 计算流/队列。

<div style="background-color: lightblue; padding: 1px;">  返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_BAD_PARAM`], [`INFINI_STATUS_INSUFFICIENT_WORKSPACE`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`], [`INFINI_STATUS_INTERNAL_ERROR`]，[`INFINI_STATUS_BAD_TENSOR_DTYPE`].

---

### 创建算子描述

```c
infiniStatus_t infiniopCreateGatherDescriptor(
    infiniopHandle_t handle,
    infiniopGatherDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t input_desc,
    infiniopTensorDescriptor_t index_desc,
    size_t dim
);
```
<div style="background-color: lightblue; padding: 1px;"> 参数：</div>

- `handle`:
  `infiniopHandle_t` 类型的硬件控柄。详情请看：[`InfiniopHandle_t`]。
- `desc_ptr`:
  `infiniopAddDescriptor_t` 指针，指向将被初始化的算子描述符地址；
- `output_desc` - { dT | ($\rm{d_o1}$, ..., $\rm{d_on}$) | ($...$) }:
     算子计算参数 `output` 的张量描述。
- `input_desc` - { dT | ($\rm{d_i1}$, ..., $\rm{d_in}$) | ($...$) }:
     算子计算参数 `input` 的张量描述。
- `index_desc` - { int | ($\rm{d_o1}$, ..., $\rm{d_on}$) | ($...$) }:
     算子计算参数 `index` 的张量描述，每一个元素值 i 应满足 0 $\leq$ i $\lt$ input_shape[dim]。
- `dim` - int:
    算子针对的维度， 满足 0 $\leq$ dim $\lt$ n。

参数限制：

- **`dT`**:  (`Float16`, `Float32`, `Bfloat16`) 之一
- `output`与`index`形状一致，且对于除`dim`的每一维度d, 满足$\rm{d_o}$[d] $\leq$ $\rm{d_i}$[d]。

<div style="background-color: lightblue; padding: 1px;"> 返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_BAD_PARAM`], [`INFINI_STATUS_BAD_TENSOR_SHAPE`], [`INFINI_STATUS_BAD_TENSOR_DTYPE`], [`INFINI_STATUS_BAD_TENSOR_STRIDES`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`].
---

### 计算额外工作空间

```c
infiniStatus_t infiniopGetGatherWorkspaceSize(
    infiniopGatherDescriptor_t desc,
    size_t *size
);
```
<div style="background-color: lightblue; padding: 1px;"> 参数：</div>

- `desc`: 使用 `infiniopCreateGatherDescriptor()` 初始化的算子描述符。
- `size`:
  额外空间大小的计算结果的写入地址；

<div style="background-color: lightblue; padding: 1px;"> 返回值：</div>

 - [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_BAD_PARAM`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`].

---

### 销毁算子描述符

```c
infiniopStatus_t infiniopDestroyGatherDescriptor(
    infiniopGatherDescriptor_t desc
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
