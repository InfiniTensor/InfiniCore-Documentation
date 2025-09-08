# `Linear Backward`

`Linear Backward`，即**线性变换反向传播**算子。该算子计算线性变换的梯度，用于神经网络的反向传播过程。

给定前向传播的计算公式：
$$ y = x \cdot w^T + b $$

反向传播计算以下梯度：
- 对输入 `x` 的梯度：$$ \frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \cdot w $$
- 对权重 `w` 的梯度：$$ \frac{\partial L}{\partial w} = \frac{\partial L}{\partial y}^T \cdot x $$
- 对偏置 `b` 的梯度：$$ \frac{\partial L}{\partial b} = \sum \frac{\partial L}{\partial y} $$

其中：
- `grad_output` 为输出梯度张量 $\frac{\partial L}{\partial y}$。
- `input` 为前向传播的输入张量 `x`。
- `weight` 为前向传播的权重张量 `w`。
- `grad_input`、`grad_weight`、`grad_bias` 分别为对应的梯度输出。

参考 `torch.nn.functional.linear` 的反向传播实现。

## 接口

### 计算

```c
infiniStatus_t infiniopLinearBackward(
    infiniopLinearBackwardDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *grad_x,
    void *grad_w,
    void *grad_b,
    const void *grad_y,
    const void *x,
    const void *w,
    void *stream
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数： </div>

- `desc`:
  已使用 `infiniopCreateLinearBackwardDescriptor()` 初始化的算子描述符；
- `workspace`:
  指向算子计算所需的额外工作空间；
- `workspace_size`:
  `workspace` 的大小，单位：字节；
- `grad_x`:
  输入梯度张量，可为 `NULL`。张量限制见[创建算子描述](#创建算子描述)部分；
- `grad_w`:
  权重梯度张量，可为 `NULL`。张量限制见[创建算子描述](#创建算子描述)部分；
- `grad_b`:
  偏置梯度张量，可为 `NULL`。张量限制见[创建算子描述](#创建算子描述)部分；
- `grad_y`:
  输出梯度张量。张量限制见[创建算子描述](#创建算子描述)部分；
- `x`:
  前向传播的输入张量。张量限制见[创建算子描述](#创建算子描述)部分；
- `w`:
  前向传播的权重张量。张量限制见[创建算子描述](#创建算子描述)部分；
- `stream`:
  计算流/队列；

<div style="background-color: lightblue; padding: 1px;"> 返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_BAD_PARAM`], [`INFINI_STATUS_INSUFFICIENT_WORKSPACE`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`], [`INFINI_STATUS_INTERNAL_ERROR`]，[`INFINI_STATUS_BAD_TENSOR_DTYPE`].

### 创建算子描述

```c
infiniStatus_t infiniopCreateLinearBackwardDescriptor(
    infiniopHandle_t handle,
    infiniopLinearBackwardDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t grad_y_desc,
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t w_desc,
    infiniopTensorDescriptor_t grad_x_desc,
    infiniopTensorDescriptor_t grad_w_desc,
    infiniopTensorDescriptor_t grad_b_desc
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数：</div>

- `handle`:
  `infiniopHandle_t` 类型的硬件控柄。详情请看：[`InfiniopHandle_t`]。
- `desc_ptr`:
  `infiniopLinearBackwardDescriptor_t` 指针，指向将被初始化的算子描述符地址；
- `grad_y_desc` - { dT | (..., out_features) | (...) }:
  算子计算参数 `grad_output` 的张量描述。
- `x_desc` - { dT | (..., in_features) | (...) }:
  算子计算参数 `input` 的张量描述。
- `w_desc` - { dT | (out_features, in_features) | (...) }:
  算子计算参数 `weight` 的张量描述。
- `grad_x_desc` - { dT | (..., in_features) | (...) }:
  算子计算参数 `grad_input` 的张量描述，可为 `NULL`。
- `grad_w_desc` - { dT | (out_features, in_features) | (...) }:
  算子计算参数 `grad_weight` 的张量描述，可为 `NULL`。
- `grad_b_desc` - { dT | (out_features,) | (...) }:
  算子计算参数 `grad_bias` 的张量描述，可为 `NULL`。

参数限制：

- `dT`: (`Float16`, `Float32`, `BFloat16`) 之一。
- `weight` 权重张量的数据为2D。
- `grad_bias` 偏置梯度张量的数据为1D，可为 `NULL`。
- 输入和输出张量的维度必须与前向传播保持一致。
- 至少需要计算一个梯度（`grad_input`、`grad_weight`、`grad_bias` 不能全为 `NULL`）。

<div style="background-color: lightblue; padding: 1px;"> 返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_BAD_PARAM`], [`INFINI_STATUS_BAD_TENSOR_SHAPE`], [`INFINI_STATUS_BAD_TENSOR_DTYPE`], [`INFINI_STATUS_BAD_TENSOR_STRIDES`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`].

### 计算额外工作空间

```c
infiniStatus_t infiniopGetLinearBackwardWorkspaceSize(
    infiniopLinearBackwardDescriptor_t desc,
    size_t *size
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数：</div>

- `desc`:
  已使用 `infiniopCreateLinearBackwardDescriptor()` 初始化的算子描述符；
- `size`:
  额外空间大小的计算结果的写入地址；

<div style="background-color: lightblue; padding: 1px;"> 返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_NULL_POINTER`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`].

### 销毁算子描述符

```c
infiniStatus_t infiniopDestroyLinearBackwardDescriptor(
    infiniopLinearBackwardDescriptor_t desc
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