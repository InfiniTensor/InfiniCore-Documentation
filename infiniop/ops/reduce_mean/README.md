# `ReduceMean`

在给定维度上计算张量元素的**平均值**。对长度为 $N$ 的一维张量 $x$ ，当在维度 `dim = 0` 上规约时：

$$
y_0=\frac{1}{N}\sum_{i=0}^{N-1} x_i
$$

对多维输入 $x\in \mathbb{R}^{d_1\times\cdots\times d_r}$，在维度 `dim = k`（$0\le k<r$）上进行规约，并**保留规约维度**（keepdim），其大小置为 1：

$$
y_{i_1,\ldots,i_{k-1},\,0,\,i_{k+1},\ldots,i_r}
=\frac{1}{d_k}\sum_{j=0}^{d_k-1} x_{i_1,\ldots,i_{k-1},\,j,\,i_{k+1},\ldots,i_r}
$$

功能参考 `torch.mean`（按给定维度取均值；仅返回**值**）。

## 接口

### 计算

```c
infiniStatus_t infiniopReduceMean(
    infiniopReduceMeanDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *input,
    void *stream);
```

<div style="background-color: lightblue; padding: 1px;"> 参数： </div>

- `desc`：已使用 `infiniopCreateReduceMeanDescriptor()` 初始化的算子描述符。
- `workspace`：指向算子计算所需的额外工作空间。
- `workspace_size`：`workspace` 的大小，单位：字节。
- `output`：输出指针。其形状需与 `input` 相同，但在 `dim` 位置大小为 1。
- `input`：输入指针。
- `stream`：计算流/队列。

<div style="background-color: lightblue; padding: 1px;"> 返回值： </div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_BAD_PARAM`], [`INFINI_STATUS_INSUFFICIENT_WORKSPACE`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`], [`INFINI_STATUS_INTERNAL_ERROR`], [`INFINI_STATUS_BAD_TENSOR_DTYPE`]


### 创建算子描述

```c
infiniStatus_t infiniopCreateReduceMeanDescriptor(
    infiniopHandle_t handle,
    infiniopReduceMeanDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t input_desc,
    size_t dim
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数： </div>

- `handle`：硬件控柄。详情请看 [`InfiniopHandle_t`]。
- `desc_ptr`：`infiniopReduceMeanDescriptor_t` 指针，指向将被初始化的算子描述符地址。
- `output_desc` - { dT | (d1, …, dr) | (…)}：输出张量描述。其秩与 `input` 相同，且在 `dim` 维度大小为 1，其余维度与 `input` 对应相等。
- `input_desc` - { dT | (d1, …, dr) | (…)}：输入张量描述。
- `dim`：规约维度，类型为 `size_t` 常量；取值范围为 $[0, r-1]$，其中 $r$ 为 `input` 的秩。

参数限制：

- `dT`：(`Float16`, `Float32`, `BFloat16`) 之一。
- `dim` 必须小于 `input` 的秩；被规约维度大小需 $\ge 1$。
- 不支持原位计算，`input` 与 `output` 不能是同一缓冲区或存在内存重叠。

<div style="background-color: lightblue; padding: 1px;"> 返回值： </div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_BAD_PARAM`], [`INFINI_STATUS_BAD_TENSOR_SHAPE`], [`INFINI_STATUS_BAD_TENSOR_DTYPE`], [`INFINI_STATUS_BAD_TENSOR_STRIDES`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`]

### 计算额外工作空间

```c
infiniStatus_t infiniopGetReduceMeanWorkspaceSize(
    infiniopReduceMeanDescriptor_t desc,
    size_t *size
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数： </div>

- `desc`：已使用 `infiniopCreateReduceMeanDescriptor()` 初始化的算子描述符。
- `size`：额外空间大小计算结果的写入地址。

<div style="background-color: lightblue; padding: 1px;"> 返回值： </div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_NULL_POINTER`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`]

### 销毁算子描述符

```c
infiniStatus_t infiniopDestroyReduceMeanDescriptor(
    infiniopReduceMeanDescriptor_t desc
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数： </div>

- `desc`：待销毁的算子描述符。

<div style="background-color: lightblue; padding: 1px;"> 返回值： </div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`]

## 已知问题

暂无

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
