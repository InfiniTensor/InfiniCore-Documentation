# `Where`

`Where` 算子为三目元素选择算子，其计算可表示为：

$$
c_i = \text{cond}_i ? a_i : b_i
$$

其中 `cond` 为条件张量，`a` 和 `b` 为输入张量，`c` 为输出张量。

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
    void *stream);
```

 参数：

- `desc`
  已使用 `infiniopCreateWhereDescriptor()` 初始化的算子描述符；
- `workspace`
  指向算子计算所需的额外工作空间；
- `workspace_size`
  `workspace` 的大小，单位：字节；
- `c`
  输出张量。张量限制见创建算子描述部分；
- `condition`
  条件张量。张量限制见创建算子描述部分；
- `a`
  输入张量。张量限制见创建算子描述部分；
- `b`
  输入张量。张量限制见创建算子描述部分；
- `stream`
  计算流/队列；

返回值：

- `INFINI_STATUS_SUCCESS`, `INFINI_STATUS_BAD_PARAM`, `INFINI_STATUS_INSUFFICIENT_WORKSPACE`, `INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`, `INFINI_STATUS_INTERNAL_ERROR`, `INFINI_STATUS_BAD_TENSOR_DTYPE`.

### 创建算子描述

```c
infiniStatus_t infiniopCreateWhereDescriptor(
    infiniopHandle_t handle,
    infiniopWhereDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t c_desc,
    infiniopTensorDescriptor_t a_desc,
    infiniopTensorDescriptor_t b_desc,
    infiniopTensorDescriptor_t condition_desc);
```

 参数： 

- `handle`
  `infiniopHandle_t` 类型的硬件句柄。详情请参见 `InfiniopHandle_t`。
- `desc_ptr`
  `infiniopWhereDescriptor_t` 指针，指向将被初始化的算子描述符地址；
- `c_desc - { dT | (d1,…,dn) | (…) }`
  算子输出张量 `c` 的张量描述，支持原位计算；
- `condition_desc- { Bool | (d1,…,dn) | (…) }`
  条件张量 `condition` 的张量描述，支持多向广播；
- `a_desc- { dT | (d1,…,dn) | (…) }`
  输入张量 `a` 的张量描述，支持原位计算，支持多向广播；
- `b_desc- { dT | (d1,…,dn) | (…) }`
  输入张量 `b` 的张量描述，支持原位计算，支持多向广播。

参数限制：

- `dT` ∈ { `int8`,`int16`,`int32`,`int164`,`Float16`, `Float32`, `Float64`, `BFloat16`,`BOOL`}；
- `cond` 的数据类型为 `Bool`，其形状需与 `a`、`b` 以及 `c` 通过多向广播后得到的形状一致；
- 输入 `a`、`b` 必须与 `c` 的形状一致，或可通过多向广播匹配 `c`；
- `condition`、`a`、`b` 的步长需与多向广播后的映射关系一致；
- 支持原位计算：`c` 可与 `a` 或 `b` 指向同一地址，但不可与 `condition` 共址；
- 输出张量 `c` 不能进行广播（`c` 的步长不能含 0）。

返回值： 

- `INFINI_STATUS_SUCCESS`, `INFINI_STATUS_BAD_PARAM`, `INFINI_STATUS_BAD_TENSOR_SHAPE`, `INFINI_STATUS_BAD_TENSOR_DTYPE`, `INFINI_STATUS_BAD_TENSOR_STRIDES`, `INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`.

### 计算额外工作空间

```c
infiniStatus_t infiniopGetWhereWorkspaceSize(
    infiniopWhereDescriptor_t desc,
    size_t *size
);
```

 参数： 

- `desc`
  已使用 `infiniopCreateWhereDescriptor()` 初始化的算子描述符；
- `size`
  额外空间大小的计算结果写入地址；

回值：

- `INFINI_STATUS_SUCCESS`, `INFINI_STATUS_NULL_POINTER`, `INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`.

### 销毁算子描述符

```c
infiniStatus_t infiniopDestroyWhereDescriptor(
    infiniopWhereDescriptor_t desc
);
```

参数：

- `desc`
  待销毁的算子描述符；

返回值： 

- `INFINI_STATUS_SUCCESS`, `INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`.

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
