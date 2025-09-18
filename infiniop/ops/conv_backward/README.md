# `ConvBackward`

卷积（Convolution）反向算子，常用于深度学习网络的反向传播阶段，实现输入、权重和偏置的梯度计算。支持普通卷积的反向传播（含可选偏置）。

## 接口

### 计算

```c
infiniStatus_t infiniopConvBackward(
    infiniopConvBackwardDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *grad_input,
    void *grad_weight,
    void *grad_bias,
    const void *grad_output,
    const void *input,
    const void *weight,
    void *stream
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数： </div>

- `desc`：已用 `infiniopCreateConvBackwardDescriptor()` 初始化的卷积反向算子描述符。
- `workspace`：临时工作空间指针。
- `workspace_size`：工作空间字节数。
- `grad_input`：输入梯度指针（输出），与前向 `input` 形状一致。
- `grad_weight`：权重梯度指针（输出），与前向 `weight` 形状一致。
- `grad_bias`：偏置梯度指针（输出），与前向 `bias` 形状一致（如有偏置）。
- `grad_output`：输出梯度指针（输入），与前向 `output` 形状一致。
- `input`：前向输入张量指针。
- `weight`：前向权重张量指针。
- `stream`：计算流/队列。

<div style="background-color: lightblue; padding: 1px;">  返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_BAD_PARAM`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`],[`INFINI_STATUS_INSUFFICIENT_WORKSPACE`]

---

### 获取卷积反向临时工作空间需求

```c
infiniStatus_t infiniopGetConvBackwardWorkspaceSize(
    infiniopConvBackwardDescriptor_t desc,
    size_t *size
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数：</div>

- `desc`：已创建的卷积反向算子描述符。
- `size`：指向 `size_t` 的指针，返回所需的工作空间字节数。

<div style="background-color: lightblue; padding: 1px;"> 返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_BAD_PARAM`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`]

---

### 创建卷积反向算子描述符

```c
infiniStatus_t infiniopCreateConvBackwardDescriptor(
    infiniopHandle_t handle,
    infiniopConvBackwardDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t grad_output_desc,
    infiniopTensorDescriptor_t input_desc,
    infiniopTensorDescriptor_t weight_desc,
    infiniopTensorDescriptor_t bias_desc,
    void *pads,
    void *strides,
    void *dilations,
    size_t n
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数：</div>

- `handle`: 硬件控柄。详情请见 [`InfiniopHandle_t`]
- `desc_ptr`: `infiniopConvBackwardDescriptor_t` 指针，指向将被初始化的算子描述符地址。
- `grad_output_desc`：输出梯度张量描述。
- `input_desc`：前向输入张量描述。
- `weight_desc`：前向权重张量描述。
- `bias_desc`：前向偏置张量描述。
- `pads`：填充指针。
- `strides`：步长指针。
- `dilations`：扩张（dilation）指针。
- `n`：卷积分组数目。

参数限制：

- **`dT`**:  (`Float16`, `Float32`, `BFloat16`) 之一。

<div style="background-color: lightblue; padding: 1px;"> 返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_BAD_PARAM`],  [`INFINI_STATUS_BAD_TENSOR_SHAPE`], [`INFINI_STATUS_BAD_TENSOR_DTYPE`], [`INFINI_STATUS_BAD_TENSOR_STRIDES`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`].

---

### 销毁卷积反向算子描述符

```c
infiniStatus_t infiniopDestroyConvBackwardDescriptor(
    infiniopConvBackwardDescriptor_t desc
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数： </div>

- `desc`
     : 待销毁的卷积反向算子描述符。

<div style="background-color: lightblue; padding: 1px;"> 返回值： </div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`].


<!-- 链接 -->
[`InfiniopHandle_t`]: /infiniop/handle/README.md

[`INFINI_STATUS_SUCCESS`]: /common/status/README.md#INFINI_STATUS_SUCCESS
[`INFINI_STATUS_BAD_PARAM`]: /common/status/README.md#INFINI_STATUS_BAD_PARAM
[`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`]: /common/status/README.md#INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED
[`INFINI_STATUS_BAD_TENSOR_SHAPE`]: /common/status/README.md#INFINI_STATUS_BAD_TENSOR_SHAPE
[`INFINI_STATUS_BAD_TENSOR_DTYPE`]: /common/status/README.md#INFINI_STATUS_BAD_TENSOR_DTYPE
[`INFINI_STATUS_BAD_TENSOR_STRIDES`]: /common/status/README.md#INFINI_STATUS_BAD_TENSOR_STRIDES
[`INFINI_STATUS_INSUFFICIENT_WORKSPACE`]: /common/status/README.md#INFINI_STATUS_INSUFFICIENT_WORKSPACE
