# `MaxPool`

最大池化（Max Pooling）算子，常用于特征提取和下采样操作。对于输入张量 $x$，在指定窗口大小、步长和填充方式下，输出张量 $y$ 的每个元素为对应窗口内的最大值。

## 接口

### 计算

```c
infiniStatus_t infiniopMaxPool(
    infiniopMaxPoolDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *input,
    void *stream
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数： </div>

- `desc`：已使用 `infiniopCreateMaxPoolDescriptor()` 初始化的最大池化算子描述符。
- `workspace`：临时工作空间指针。
- `workspace_size`：工作空间字节数。
- `output`：输出指针。
- `input`：输入指针。
- `stream`：计算流/队列。

<div style="background-color: lightblue; padding: 1px;">  返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_BAD_PARAM`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`]

---

### 获取最大池化临时工作空间需求

```c
infiniStatus_t infiniopGetMaxPoolWorkspaceSize(
    infiniopMaxPoolDescriptor_t desc,
    size_t *size
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数：</div>

- `desc`：已创建的最大池化算子描述符。
- `size`：指向 `size_t` 的指针，返回所需的工作空间字节数。

<div style="background-color: lightblue; padding: 1px;"> 返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_BAD_PARAM`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`]

---

### 创建最大池化算子描述符

```c
infiniStatus_t infiniopCreateMaxPoolDescriptor(
    infiniopHandle_t handle,
    infiniopMaxPoolDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t input_desc,
    void *kernel_size,
    void *strides,
    void *pads,
    bool ceil_mode
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数：</div>

- `handle`: 硬件控柄。详情请见 [`InfiniopHandle_t`]
- `desc_ptr`: `infiniopMaxPoolDescriptor_t` 指针，指向将被初始化的算子描述符地址。
- `output_desc`：输出张量描述。
- `input_desc`：输入张量描述，张量形状与 `output_desc`、池化参数共同决定。
- `kernel_size`：窗口尺寸指针（如 `{kh, kw}`），类型与输入张量一致，需为正整数数组。
- `strides`：步长指针（如 `{sh, sw}`），类型与输入张量一致，需为正整数数组。
- `pads`：填充指针（如 `{ph, pw}`），类型与输入张量一致，需为非负整数数组。
- `ceil_mode`：是否采用上取整（`true` 采用 ceil，`false` 采用 floor）。

参数限制：

- **`dT`**:  (`Float16`, `Float32`, `BFloat16`) 之一。

<div style="background-color: lightblue; padding: 1px;"> 返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_BAD_PARAM`],  [`INFINI_STATUS_BAD_TENSOR_SHAPE`], [`INFINI_STATUS_BAD_TENSOR_DTYPE`], [`INFINI_STATUS_BAD_TENSOR_STRIDES`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`].

---

### 销毁最大池化算子描述符

```c
infiniStatus_t infiniopDestroyMaxPoolDescriptor(
    infiniopMaxPoolDescriptor_t desc
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数： </div>

- `desc`
     : 待销毁的最大池化算子描述符。

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