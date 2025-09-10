# `Cast`

`Cast`（类型转换算子）为**单目逐元素算子**。用于在指定的数据类型之间进行转换：

- **整数类型互转**：`int32`、`int64`、`uint32`、`uint64`；
- **浮点类型互转**：`f64`、`f32`、`f16`、`bf16`；
- **整数与浮点互转**：`int32/int64/uint32/uint64` ↔ `f64/f32/f16/bf16`；

`input` 为输入，`output` 为输出。，数据按元素一一映射。
## 接口

### 计算

```c
infiniStatus_t infiniopCast(
    infiniopCastDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *input,
    void *stream
);
```

<div style="background-color: lightblue; padding: 1px;">参数：</div>

- `desc`：
  已使用 `infiniopCreateCastDescriptor()` 初始化的算子描述符；
- `workspace`：
  指向算子计算所需的额外工作空间；
- `workspace_size`：
  `workspace` 的大小，单位：字节；
- `output`：
  输出张量（转换后的数据）。张量限制见[创建算子描述](#创建算子描述)部分；
- `input`：
  输入张量（待转换的数据）。张量限制见[创建算子描述](#创建算子描述)部分；
- `stream`：
  计算流/队列；

<div style="background-color: lightblue; padding: 1px;">返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_BAD_PARAM`], [`INFINI_STATUS_INSUFFICIENT_WORKSPACE`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`], [`INFINI_STATUS_INTERNAL_ERROR`], [`INFINI_STATUS_BAD_TENSOR_DTYPE`].

### 创建算子描述

```c
infiniStatus_t infiniopCreateCastDescriptor(
    infiniopHandle_t handle,
    infiniopCastDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t input_desc
);
```

<div style="background-color: lightblue; padding: 1px;">参数：</div>

- `handle`：
  `infiniopHandle_t` 类型的硬件控柄。详情请看：[`InfiniopHandle_t`]。
- `desc_ptr`：
  `infiniopCastDescriptor_t` 指针，指向将被初始化的算子描述符地址；
- `output_desc` - { dT | (d1,...,dn) | (...) }：
  参数 `output` 的张量描述（目标数据类型由此指定）。**不支持原位计算**；
- `input_desc` - { dT | (d1,...,dn) | (...) }：
  参数 `input` 的张量描述。

参数限制：

- `dT` 支持：

  - **整数**：`Int32`、`Int64`、`UInt32`、`UInt64`；
  - **浮点**：`Float64`、`Float32`、`Float16`、`BFloat16`。
- **允许的类型组合**：

  1. 整数 ↔ 整数（上述整数集合内互转）；
  2. 浮点 ↔ 浮点（上述浮点集合内互转）；
  3. 整数 ↔ 浮点；
- `output` 与 `input` 的形状与步长需对应（逐元素一一映射）；不涉及广播；`output` 的步长不得为 0。
- 不支持 inplace：`output` 不能与 `input` 指向同一地址。

> 注：转换遵循常规数值转换语义（IEEE 754 浮点到浮点、整数到浮点的表示变换）。当整数值超出目标浮点的精确表示范围时，将发生舍入。

<div style="background-color: lightblue; padding: 1px;">返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_BAD_PARAM`], [`INFINI_STATUS_BAD_TENSOR_SHAPE`], [`INFINI_STATUS_BAD_TENSOR_DTYPE`], [`INFINI_STATUS_BAD_TENSOR_STRIDES`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`].

### 计算额外工作空间

```c
infiniStatus_t infiniopGetCastWorkspaceSize(
    infiniopCastDescriptor_t desc,
    size_t *size
);
```

<div style="background-color: lightblue; padding: 1px;">参数：</div>

- `desc`：
  已使用 `infiniopCreateCastDescriptor()` 初始化的算子描述符；
- `size`：
  额外空间大小的计算结果的写入地址；

<div style="background-color: lightblue; padding: 1px;">返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_NULL_POINTER`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`].

### 销毁算子描述符

```c
infiniStatus_t infiniopDestroyCastDescriptor(
    infiniopCastDescriptor_t desc
);
```

<div style="background-color: lightblue; padding: 1px;">参数：</div>

- `desc`：
  输入。待销毁的算子描述符；

<div style="background-color: lightblue; padding: 1px;">返回值：</div>

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
