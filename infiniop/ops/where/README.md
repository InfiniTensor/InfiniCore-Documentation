# `Where`

`Where`, 即**条件选择**算子，为三目逐元素算子。其计算可被表述为：

$$ c[i] = \begin{cases} 
a[i] & \text{if } condition[i] \text{ is true} \\
b[i] & \text{if } condition[i] \text{ is false}
\end{cases} $$

其中 `condition` 为条件张量（布尔类型），`a` 和 `b` 为输入张量，`c` 为输出张量。

## 接口

### 计算

```c
infiniStatus_t infiniopWhere(
    infiniopWhereDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    const void *condition,
    const void *a,
    const void *b,
    void *c,
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
- `condition`:
  条件张量（布尔类型）。张量限制见[创建算子描述](#创建算子描述)部分；
- `a`:
  输入张量a。张量限制见[创建算子描述](#创建算子描述)部分；
- `b`:
  输入张量b。张量限制见[创建算子描述](#创建算子描述)部分；
- `c`:
  输出张量。张量限制见[创建算子描述](#创建算子描述)部分；
- `stream`:
  计算流/队列；

<div style="background-color: lightblue; padding: 1px;"> 返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_BAD_PARAM`], [`INFINI_STATUS_INSUFFICIENT_WORKSPACE`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`], [`INFINI_STATUS_INTERNAL_ERROR`]，[`INFINI_STATUS_BAD_TENSOR_DTYPE`].

### 创建算子描述

```c
infiniStatus_t infiniopCreateWhereDescriptor(
    infiniopHandle_t handle,
    infiniopWhereDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t condition,
    infiniopTensorDescriptor_t a,
    infiniopTensorDescriptor_t b,
    infiniopTensorDescriptor_t c
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数：</div>

- `handle`:
  `infiniopHandle_t` 类型的硬件控柄。详情请看：[`InfiniopHandle_t`]。
- `desc_ptr`:
  `infiniopWhereDescriptor_t` 指针，指向将被初始化的算子描述符地址；
- `condition` - { Bool | (d1,...,dn) | (...) }:
  算子计算参数 `condition` 的张量描述。
- `a` - { dT | (d1,...,dn) | (...) }:
  算子计算参数 `a` 的张量描述，支持原位计算。
- `b` - { dT | (d1,...,dn) | (...) }:
  算子计算参数 `b` 的张量描述，支持原位计算。
- `c` - { dT | (d1,...,dn) | (...) }:
  算子计算参数 `c` 的张量描述，支持原位计算。

参数限制：

- `dT`:  (`Float16`, `Float32`, `BFloat16`) 之一。
- `condition` 的数据类型必须为 `Bool`。
- 所有张量 `condition`、`a`、`b` 和 `c` 的形状需相同。
- 支持原位计算，即计算时 `c` 可以和 `a` 或 `b` 指向同一地址。
- 输入输出类型一致（`a`、`b`、`c` 类型相同）。

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