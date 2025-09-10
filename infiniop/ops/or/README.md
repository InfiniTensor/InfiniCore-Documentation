# Or 算子

Or 算子用于执行张量的逐元素逻辑或运算。该算子参考 `torch.logical_or` 的行为。

## 数学定义

给定两个布尔张量 $a$ 和 $b$，Or 算子的计算公式为：

$$c = \text{or}(a, b) = a \lor b$$

其中 $c$ 是输出张量，$a$ 和 $b$ 是输入张量。该算子支持多向广播。

## 接口

### 计算

```c
infiniStatus_t infiniopOr(
    infiniopOrDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *c,
    const void *a,
    const void *b,
    void *stream
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数：</div>

- `desc`:
  已使用 `infiniopCreateOrDescriptor()` 初始化的算子描述符。
- `workspace`:
  额外工作空间地址。
- `workspace_size`:
  `workspace` 的大小，单位：字节。
- `c`:
  计算结果地址，支持原位计算。
- `a`:
  输入张量 `a` 的数据指针。
- `b`:
  输入张量 `b` 的数据指针。
- `stream`:
  计算流/队列。

<div style="background-color: lightblue; padding: 1px;"> 返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_NULL_POINTER`], [`INFINI_STATUS_INSUFFICIENT_WORKSPACE`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`], [`INFINI_STATUS_INTERNAL_ERROR`].

### 创建算子描述

```c
infiniStatus_t infiniopCreateOrDescriptor(
    infiniopHandle_t handle,
    infiniopOrDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t c,
    infiniopTensorDescriptor_t a,
    infiniopTensorDescriptor_t b
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数：</div>

- `handle`:
  硬件控柄。详见 [`InfiniopHandle_t`]。
- `desc_ptr`:
  `infiniopOrDescriptor_t` 指针，指向将被初始化的算子描述符地址。
- `c` - { BOOL | (d1,...,dn) | (...) }:
  算子计算参数 `c` 的张量描述，支持原位计算。
- `a` - { BOOL | (d1,...,dn) | (...) }:
  算子计算参数 `a` 的张量描述，支持原位计算，支持多向广播。
- `b` - { BOOL | (d1,...,dn) | (...) }:
  算子计算参数 `b` 的张量描述，支持原位计算，支持多向广播。

参数限制：

- 所有张量的数据类型必须为 `Bool`。
- 输入 `a` 与 `b` 的形状需与 `c` 相同。`a` 与 `b` 涉及多向广播时需调整步长以匹配多向广播的映射关系。
- 支持原位计算，即计算时 `c` 可以和 `a` 或 `b` 指向同一地址。
- 计算输出参数 `c` 不能进行广播（`c` 的步长不能涉及广播设置，即步长不能有 0）。

<div style="background-color: lightblue; padding: 1px;"> 返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_BAD_PARAM`], [`INFINI_STATUS_BAD_TENSOR_SHAPE`], [`INFINI_STATUS_BAD_TENSOR_DTYPE`], [`INFINI_STATUS_BAD_TENSOR_STRIDES`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`].

### 计算额外工作空间

```c
infiniStatus_t infiniopGetOrWorkspaceSize(
    infiniopOrDescriptor_t desc,
    size_t *size
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数：</div>

- `desc`:
  已使用 `infiniopCreateOrDescriptor()` 初始化的算子描述符。
- `size`:
  额外空间大小的计算结果的写入地址。

<div style="background-color: lightblue; padding: 1px;"> 返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_NULL_POINTER`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`].

### 销毁算子描述符

```c
infiniStatus_t infiniopDestroyOrDescriptor(
    infiniopOrDescriptor_t desc
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数： </div>

- `desc`:
  待销毁的算子描述符。

<div style="background-color: lightblue; padding: 1px;"> 返回值： </div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`].

## 已知问题

无

<!-- 链接 -->
[`InfiniopHandle_t`]: /infiniop/handle/README.md

[`INFINI_STATUS_SUCCESS`]: /common/status/README.md#INFINI_STATUS_SUCCESS
[`INFINI_STATUS_BAD_PARAM`]: /common/status/README.md#INFINI_STATUS_BAD_PARAM
[`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`]: /common/status/README.md#INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED
[`INFINI_STATUS_BAD_TENSOR_SHAPE`]: /common/status/README.md#INFINI_STATUS_BAD_TENSOR_SHAPE
[`INFINI_STATUS_BAD_TENSOR_DTYPE`]: /common/status/README.md#INFINI_STATUS_BAD_TENSOR_DTYPE
[`INFINI_STATUS_BAD_TENSOR_STRIDES`]: /common/status/README.md#INFINI_STATUS_BAD_TENSOR_STRIDES
[`INFINI_STATUS_NULL_POINTER`]:/common/status/README.md#INFINI_STATUS_NULL_POINTER
[`INFINI_STATUS_INSUFFICIENT_WORKSPACE`]:/common/status/README.md#INFINI_STATUS_INSUFFICIENT_WORKSPACE
[`INFINI_STATUS_INTERNAL_ERROR`]:/common/status/README.md#INFINI_STATUS_INTERNAL_ERROR