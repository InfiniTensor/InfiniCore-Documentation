# `LogSoftmax`

对数 Softmax 算子，计算输入张量的对数 softmax 值。对于长度为 $N$ 的一维张量 $x$，其公式为：

$$ y_i = \log\left(\frac{e^{x_i}}{\sum_{j=0}^{N-1} e^{x_j}}\right) = x_i - \log\left(\sum_{j=0}^{N-1} e^{x_j}\right) $$

对于多维输入的情况，则在最后一个维度上执行对数 softmax 计算。

## 接口

### 计算

```c
infiniStatus_t infiniopLogSoftmax(
    infiniopLogSoftmaxDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *y,
    const void *x,
    void *stream
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数： </div>

- `desc`: 使用 `infiniopCreateLogSoftmaxDescriptor()` 初始化的算子描述符。
- `workspace`: 算子计算所需的额外工作空间。
- `workspace_size`: `workspace` 的大小，单位：字节（byte）。
- `y`: 计算结果的数据地址。张量限制见[创建算子描述](#创建算子描述)部分。
- `x`: 输入数据地址，可以与 `y` 相同。张量限制见[创建算子描述](#创建算子描述)部分。
- `stream`: 计算流/队列。

<div style="background-color: lightblue; padding: 1px;">  返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_BAD_PARAM`], [`INFINI_STATUS_INSUFFICIENT_WORKSPACE`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`], [`INFINI_STATUS_INTERNAL_ERROR`].

---

### 创建算子描述

```c
infiniStatus_t infiniopCreateLogSoftmaxDescriptor(
    infiniopHandle_t handle,
    infiniopLogSoftmaxDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数：</div>

- `handle`: `infiniopHandle_t` 类型的硬件控柄。详情请看：[`InfiniopHandle_t`]。
- `desc_ptr`: 存放将被初始化的算子描述符的地址。
- `y_desc` - { dT | (d1,...,dn) | (...) }: 算子计算参数 `y` 的张量描述，支持2D和3D张量。
- `x_desc` - { dT | (d1,...,dn) | (...) }: 算子计算参数 `x` 的张量描述，形状与 `y_desc` 一致。

参数限制：

- **`dT`**: (`Float16`, `BFloat16`, `Float32`) 之一。
- 支持原位计算，即计算时 `y` 可以和 `x` 指向同一地址。
- 输入张量支持2D和3D形状，在最后一个维度上执行LogSoftmax计算。

<div style="background-color: lightblue; padding: 1px;"> 返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_BAD_PARAM`], [`INFINI_STATUS_BAD_TENSOR_SHAPE`], [`INFINI_STATUS_BAD_TENSOR_DTYPE`], [`INFINI_STATUS_BAD_TENSOR_STRIDES`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`].

---

### 计算额外工作空间

```c
infiniStatus_t infiniopGetLogSoftmaxWorkspaceSize(
    infiniopLogSoftmaxDescriptor_t desc,
    size_t *size
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数： </div>

- `desc`: 使用 `infiniopCreateLogSoftmaxDescriptor()` 初始化的算子描述符。
- `size`: 存放额外空间大小的计算结果的地址。

<div style="background-color: lightblue; padding: 1px;"> 返回值： </div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_BAD_PARAM`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`].

---

### 销毁算子描述符

```c
infiniStatus_t infiniopDestroyLogSoftmaxDescriptor(
    infiniopLogSoftmaxDescriptor_t desc
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数： </div>

- `desc`: 待销毁的算子描述符。

<div style="background-color: lightblue; padding: 1px;"> 返回值： </div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`].

## 已知问题

### 平台限制

- 只完成了英伟达平台的实现。

<!-- 链接 -->
[`InfiniopHandle_t`]: /infiniop/handle/README.md

[`INFINI_STATUS_SUCCESS`]: /common/status/README.md#INFINI_STATUS_SUCCESS
[`INFINI_STATUS_BAD_PARAM`]: /common/status/README.md#INFINI_STATUS_BAD_PARAM
[`INFINI_STATUS_INSUFFICIENT_WORKSPACE`]: /common/status/README.md#INFINI_STATUS_INSUFFICIENT_WORKSPACE
[`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`]: /common/status/README.md#INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED
[`INFINI_STATUS_BAD_TENSOR_DTYPE`]: /common/status/README.md#INFINI_STATUS_BAD_TENSOR_DTYPE
[`INFINI_STATUS_BAD_TENSOR_SHAPE`]: /common/status/README.md#INFINI_STATUS_BAD_TENSOR_SHAPE
[`INFINI_STATUS_BAD_TENSOR_STRIDES`]: /common/status/README.md#INFINI_STATUS_BAD_TENSOR_STRIDES
[`INFINI_STATUS_INTERNAL_ERROR`]: /common/status/README.md#INFINI_STATUS_INTERNAL_ERROR