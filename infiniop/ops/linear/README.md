# `Linear`

`Linear`，即**线性变换**算子。该算子执行线性变换操作，计算公式为：

$$ y = x \cdot w^T + b $$

其中：

- `x` 为输入张量。
- `w` 为权重张量，形状为 `(out_features, in_features)`。
- `b` 为偏置张量，形状为 `(out_features,)`，可选参数。
- `y` 为输出张量。

参考 `torch.nn.functional.linear` 实现。

## 接口

### 计算

```c
infiniStatus_t infiniopLinear(
    infiniopLinearDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *y,
    const void *x,
    const void *w,
    const void *b,
    void *stream
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数： </div>

- `desc`:
  已使用 `infiniopCreateLinearDescriptor()` 初始化的算子描述符；
- `workspace`:
  指向算子计算所需的额外工作空间；
- `workspace_size`:
  `workspace` 的大小，单位：字节；
- `y`:
  输出张量。张量限制见[创建算子描述](#创建算子描述)部分；
- `x`:
  输入张量。张量限制见[创建算子描述](#创建算子描述)部分；
- `w`:
  权重张量。张量限制见[创建算子描述](#创建算子描述)部分；
- `b`:
  偏置张量，可为 `NULL`。张量限制见[创建算子描述](#创建算子描述)部分；
- `stream`:
  计算流/队列；

<div style="background-color: lightblue; padding: 1px;"> 返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_BAD_PARAM`], [`INFINI_STATUS_INSUFFICIENT_WORKSPACE`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`], [`INFINI_STATUS_INTERNAL_ERROR`]，[`INFINI_STATUS_BAD_TENSOR_DTYPE`].

### 创建算子描述

```c
infiniStatus_t infiniopCreateLinearDescriptor(
    infiniopHandle_t handle,
    infiniopLinearDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t w_desc,
    infiniopTensorDescriptor_t b_desc,
    infiniopTensorDescriptor_t y_desc
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数：</div>

- `handle`:
  `infiniopHandle_t` 类型的硬件控柄。详情请看：[`InfiniopHandle_t`]。
- `desc_ptr`:
  `infiniopLinearDescriptor_t` 指针，指向将被初始化的算子描述符地址；
- `x_desc` - { dT | (..., in_features) | (...) }:
  算子计算参数 `x` 的张量描述。
- `w_desc` - { dT | (out_features, in_features) | (...) }:
  算子计算参数 `w` 的张量描述，权重数据为2D。
- `b_desc` - { dT | (out_features,) | (...) }:
  算子计算参数 `b` 的张量描述，偏置数据为1D，可为 `NULL`。
- `y_desc` - { dT | (..., out_features) | (...) }:
  算子计算参数 `y` 的张量描述。

参数限制：

- `dT`: (`Float16`, `Float32`, `BFloat16`) 之一。
- `w` 权重张量的数据为2D。
- `b` 偏置张量的数据为1D，需要支持不传 `bias` 的情况。
- 输入 `x` 的最后一个维度必须与权重 `w` 的第二个维度相匹配。
- 输出 `y` 的最后一个维度必须与权重 `w` 的第一个维度相匹配。

<div style="background-color: lightblue; padding: 1px;"> 返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_BAD_PARAM`], [`INFINI_STATUS_BAD_TENSOR_SHAPE`], [`INFINI_STATUS_BAD_TENSOR_DTYPE`], [`INFINI_STATUS_BAD_TENSOR_STRIDES`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`].

### 计算额外工作空间

```c
infiniStatus_t infiniopGetLinearWorkspaceSize(
    infiniopLinearDescriptor_t desc,
    size_t *size
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数：</div>

- `desc`:
  已使用 `infiniopCreateLinearDescriptor()` 初始化的算子描述符；
- `size`:
  额外空间大小的计算结果的写入地址；

<div style="background-color: lightblue; padding: 1px;"> 返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_NULL_POINTER`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`].

### 销毁算子描述符

```c
infiniStatus_t infiniopDestroyLinearDescriptor(
    infiniopLinearDescriptor_t desc
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