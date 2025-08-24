# `LeakyReLU`

`LeakyReLU`（带泄露的线性整流）为**单目逐元素激活算子**，其计算为：

$$
\mathrm{LeakyReLU}(x)=
\begin{cases}
x, & x \ge 0,\\[2pt]
\text{negative\_slope}\cdot x, & x<0.
\end{cases}
$$

其中 `input` 为输入，`output` 为输出；`negative_slope` 为算子构建时设置的标量常数（`float` 类型），对所有元素生效。

## 接口

### 计算

```c
infiniStatus_t infiniopLeakyReLU(
    infiniopLeakyReLUDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *input,
    void *stream
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数： </div>

- `desc`:
  已使用 `infiniopCreateLeakyReLUDescriptor()` 初始化的算子描述符；
- `workspace`:
  指向算子计算所需的额外工作空间；
- `workspace_size`:
  `workspace` 的大小，单位：字节；
- `output`：
  输出张量。张量限制见[创建算子描述](#创建算子描述)部分；
- `input`：
  输入张量。张量限制见[创建算子描述](#创建算子描述)部分；
- `stream`：
  计算流/队列；

<div style="background-color: lightblue; padding: 1px;"> 返回值： </div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_BAD_PARAM`], [`INFINI_STATUS_INSUFFICIENT_WORKSPACE`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`], [`INFINI_STATUS_INTERNAL_ERROR`], [`INFINI_STATUS_BAD_TENSOR_DTYPE`].

### 创建算子描述

```c
infiniStatus_t infiniopCreateLeakyReLUDescriptor(
    infiniopHandle_t handle,
    infiniopLeakyReLUDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t input_desc,
    float negative_slope
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数： </div>

- `handle`:
  `infiniopHandle_t` 类型的硬件控柄。详情请看：[`InfiniopHandle_t`]。
- `desc_ptr`:
  `infiniopLeakyReLUDescriptor_t` 指针，指向将被初始化的算子描述符地址；
- `output_desc` - { dT | (d1,...,dn) | (...) }：
  参数 `output` 的张量描述，支持原位计算；
- `input_desc` - { dT | (d1,...,dn) | (...) }：
  参数 `input` 的张量描述，支持原位计算；
- `negative_slope`:
  泄露系数，`float`（32 位）标量常数；在整个计算过程中保持不变；

参数限制：

- `dT`: (`Float64`, `Float32`, `Float16`, `BFloat16`) 之一。
- `input` 与 `output` 的数据类型必须一致。
- `output` 的形状与步长需与 `input` 对应（逐元素一一映射）；不涉及广播。
- 支持原位计算，即计算时 `output` 可以与 `input` 指向同一地址。
- 计算输出参数 `output` 不能进行广播（其步长不得为 0）。

<div style="background-color: lightblue; padding: 1px;"> 返回值： </div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_BAD_PARAM`], [`INFINI_STATUS_BAD_TENSOR_SHAPE`], [`INFINI_STATUS_BAD_TENSOR_DTYPE`], [`INFINI_STATUS_BAD_TENSOR_STRIDES`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`].

### 计算额外工作空间

```c
infiniStatus_t infiniopGetLeakyReLUWorkspaceSize(
    infiniopLeakyReLUDescriptor_t desc,
    size_t *size
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数： </div>

- `desc`:
  已使用 `infiniopCreateLeakyReLUDescriptor()` 初始化的算子描述符；
- `size`:
  额外空间大小的计算结果的写入地址；

<div style="background-color: lightblue; padding: 1px;"> 返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_NULL_POINTER`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`].

### 销毁算子描述符

```c
infiniStatus_t infiniopDestroyLeakyReLUDescriptor(
    infiniopLeakyReLUDescriptor_t desc
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数： </div>

- `desc`:
  输入。待销毁的算子描述符；

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
