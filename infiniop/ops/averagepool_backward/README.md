# `AvgPoolBackward`

平均池化（Average Pooling）反向算子，常用于反向传播阶段的梯度计算。对于输入张量 $x$ 和上游梯度 $grad\_output$，该算子根据平均池化前向操作计算输入的梯度 $grad\_input$。

## 接口

### 计算

```c
infiniStatus_t infiniopAvgPoolBackward(
    infiniopAvgPoolBackwardDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *grad_input,
    const void *grad_output,
    const void *input,
    void *stream
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数： </div>

- `desc`：已使用 `infiniopCreateAvgPoolBackwardDescriptor()` 初始化的平均池化反向算子描述符。
- `workspace`：临时工作空间指针。
- `workspace_size`：工作空间字节数。
- `grad_input`：输入梯度指针（输出）。
- `grad_output`：输出梯度指针（输入）。
- `input`：前向输入张量指针。
- `stream`：计算流/队列。

<div style="background-color: lightblue; padding: 1px;">  返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_BAD_PARAM`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`]

---

### 获取平均池化反向临时工作空间需求

```c
infiniStatus_t infiniopGetAvgPoolBackwardWorkspaceSize(
    infiniopAvgPoolBackwardDescriptor_t desc,
    size_t *size
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数：</div>

- `desc`：已创建的平均池化反向算子描述符。
- `size`：指向 `size_t` 的指针，返回所需的工作空间字节数。

<div style="background-color: lightblue; padding: 1px;"> 返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_BAD_PARAM`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`]

---

### 创建平均池化反向算子描述符

```c
infiniStatus_t infiniopCreateAvgPoolBackwardDescriptor(
    infiniopHandle_t handle,
    infiniopAvgPoolBackwardDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t grad_input_desc,
    infiniopTensorDescriptor_t grad_output_desc,
    infiniopTensorDescriptor_t input_desc,
    void *kernel_size,
    void *strides,
    void *pads,
    bool ceil_mode
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数：</div>

- `handle`: 硬件控柄。详情请见 [`InfiniopHandle_t`]
- `desc_ptr`: `infiniopAvgPoolBackwardDescriptor_t` 指针，指向将被初始化的算子描述符地址。
- `grad_input_desc`：输入梯度张量描述。
- `grad_output_desc`：输出梯度张量描述。
- `input_desc`：前向输入张量描述。
- `kernel_size`：窗口尺寸指针。
- `strides`：步长指针。
- `pads`：填充指针。
- `ceil_mode`：是否采用上取整。

参数限制：

- **`dT`**:  (`Float16`, `Float32`, `BFloat16`) 之一。

<div style="background-color: lightblue; padding: 1px;"> 返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_BAD_PARAM`],  [`INFINI_STATUS_BAD_TENSOR_SHAPE`], [`INFINI_STATUS_BAD_TENSOR_DTYPE`], [`INFINI_STATUS_BAD_TENSOR_STRIDES`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`].

---

### 销毁平均池化反向算子描述符

```c
infiniStatus_t infiniopDestroyAvgPoolBackwardDescriptor(
    infiniopAvgPoolBackwardDescriptor_t desc
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数： </div>

- `desc`
     : 待销毁的平均池化反向算子描述符。

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
