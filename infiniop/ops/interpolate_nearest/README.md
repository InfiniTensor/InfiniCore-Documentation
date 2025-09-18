# `InterpolateNearest`

最近邻插值（Nearest Neighbor Interpolation）算子，常用于特征图、图像等张量的上采样/下采样。该算子根据输入和输出张量的形状，将输入数据通过最近邻策略映射到输出张量。

## 接口

### 计算

```c
infiniStatus_t infiniopInterpolateNearest(
    infiniopInterpolateNearestDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *input,
    void *stream
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数： </div>

- `desc`：已用 `infiniopCreateInterpolateNearestDescriptor()` 初始化的最近邻插值算子描述符。
- `workspace`：临时工作空间指针。
- `workspace_size`：工作空间字节数。
- `output`：输出张量指针。
- `input`：输入张量指针。
- `stream`：计算流/队列。

<div style="background-color: lightblue; padding: 1px;">  返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_BAD_PARAM`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`]

---

### 获取最近邻插值临时工作空间需求

```c
infiniStatus_t infiniopGetInterpolateNearestWorkspaceSize(
    infiniopInterpolateNearestDescriptor_t desc,
    size_t *size
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数：</div>

- `desc`：已创建的最近邻插值算子描述符。
- `size`：指向 `size_t` 的指针，返回所需的工作空间字节数。

<div style="background-color: lightblue; padding: 1px;"> 返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_BAD_PARAM`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`]

---

### 创建最近邻插值算子描述符

```c
infiniStatus_t infiniopCreateInterpolateNearestDescriptor(
    infiniopHandle_t handle,
    infiniopInterpolateNearestDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t input_desc
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数：</div>

- `handle`: 硬件控柄。详情请见 [`InfiniopHandle_t`]
- `desc_ptr`: `infiniopInterpolateNearestDescriptor_t` 指针，指向将被初始化的算子描述符地址。
- `output_desc`：输出张量描述。
- `input_desc`：输入张量描述。

参数限制：

- **`dT`**:  (`Float16`, `Float32`, `BFloat16`) 之一。

<div style="background-color: lightblue; padding: 1px;"> 返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_BAD_PARAM`],  [`INFINI_STATUS_BAD_TENSOR_SHAPE`], [`INFINI_STATUS_BAD_TENSOR_DTYPE`], [`INFINI_STATUS_BAD_TENSOR_STRIDES`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`].

---

### 销毁最近邻插值算子描述符

```c
infiniStatus_t infiniopDestroyInterpolateNearestDescriptor(
    infiniopInterpolateNearestDescriptor_t desc
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数： </div>

- `desc`
     : 待销毁的最近邻插值算子描述符。

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