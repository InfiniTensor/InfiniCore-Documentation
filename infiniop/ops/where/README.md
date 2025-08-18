
# `Where`

`Where`，即**条件选择**算子，为三目逐元素算子。其计算可被表述为：

$$
c = \text{condition} \ ? \ a : b
$$

其中 `a`、`b` 与 `condition` 为输入，`c` 为输出。`condition` 为判定条件。

## 接口

### 计算

```c
infiniStatus_t infiniopWhere(
    infiniopWhereDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *c,
    const void *a,
    const void *b,
    const void *condition,
    void *stream
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数： </div>

- `desc`:
  已使用 `infiniopCreateWhereDescriptor()` 初始化的算子描述符；
- `workspace`:
  指向算子计算所需的额外工作空间；
- `workspace_size`:
  `workspace` 的大小，单位：字节；
- `c`:
  输出张量。张量限制见[创建算子描述](#创建算子描述)部分；
- `a`:
  输入张量。张量限制见[创建算子描述](#创建算子描述)部分；
- `b`:
  输入张量。张量限制见[创建算子描述](#创建算子描述)部分；
- `condition`:
  条件张量。张量限制见[创建算子描述](#创建算子描述)部分； 
- `stream`:
  计算流/队列；

<div style="background-color: lightblue; padding: 1px;"> 返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_BAD_PARAM`], [`INFINI_STATUS_INSUFFICIENT_WORKSPACE`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`], [`INFINI_STATUS_INTERNAL_ERROR`]，[`INFINI_STATUS_BAD_TENSOR_DTYPE`].

### 创建算子描述

```c
infiniStatus_t infiniopCreateWhereDescriptor(
    infiniopHandle_t handle,
    infiniopWhereDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t c_desc,
    infiniopTensorDescriptor_t a_desc,
    infiniopTensorDescriptor_t b_desc,
    infiniopTensorDescriptor_t condition_desc
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数：</div>

- `handle`:
  `infiniopHandle_t` 类型的硬件控柄。详情请看：[`InfiniopHandle_t`]。
- `desc_ptr`:
  `infiniopWhereDescriptor_t` 指针，指向将被初始化的算子描述符地址；
- `c_desc` - { dT | (d1,...,dn) | (...) }:
  算子计算参数 `c` 的张量描述，支持原位计算。
- `a_desc` - { dT | (d1,...,dn) | (...) }:
  算子计算参数 `a` 的张量描述，支持原位计算，支持多向广播。
- `b_desc` - { dT | (d1,...,dn) | (...) }:
  算子计算参数 `b` 的张量描述，支持原位计算，支持多向广播。
- `condition_desc` - { Bool | (d1,...,dn) | (...) }:
  算子计算参数 `condition` 的张量描述，支持原位计算，支持多向广播。

参数限制：

- `dT`:  (`Float16`, `Float32`, `Float64`, `BFloat16`, `Int8`, `Int16`, `Int32`, `Int64`, `UInt8`, `UInt16`, `UInt32`, `UInt64`) 之一。
- 输入 `a` 、 `b` 、 `condition` 的形状需与 `c` 相同。`a` 、 `b` 、 `condition` 涉及多向广播时需调整步长以匹配多向广播的映射关系。
- 支持原位计算，即计算时 `c` 可以和 `a` 或 `b` 或 `condition` 指向同一地址。
- 计算输出参数 `c` 不能进行广播（`c` 的步长不能涉及广播设置，即步长不能有 0）

<div style="background-color: lightblue; padding: 1px;"> 返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_BAD_PARAM`], [`INFINI_STATUS_BAD_TENSOR_SHAPE`], [`INFINI_STATUS_BAD_TENSOR_DTYPE`], [`INFINI_STATUS_BAD_TENSOR_STRIDES`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`].

### 计算额外工作空间

```c
infiniStatus_t infiniopGetWhereWorkspaceSize(
    infiniopWhereDescriptor_t desc,
    size_t *size
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数：</div>

- `desc`:
  已使用 `infiniopCreateWhereDescriptor()` 初始化的算子描述符；
- `size`:
  额外空间大小的计算结果的写入地址；

<div style="background-color: lightblue; padding: 1px;"> 返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_NULL_POINTER`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`].

### 销毁算子描述符

```c
infiniStatus_t infiniopDestroyWhereDescriptor(
    infiniopWhereDescriptor_t desc
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
