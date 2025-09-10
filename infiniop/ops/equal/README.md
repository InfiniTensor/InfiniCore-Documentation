# Equal 算子

Equal 算子比较两个输入张量是否完全相等，返回一个布尔标量。该算子参考 `torch.equal` 的行为。

## 数学定义

给定两个张量 $a$ 和 $b$，Equal 算子的计算公式为：

$$c = \text{equal}(a, b) = \begin{cases}
\text{true} & \text{if } a = b \text{ (形状和所有元素都相同)} \\
\text{false} & \text{otherwise}
\end{cases}$$

其中 $c$ 是输出的布尔标量，$a$ 和 $b$ 是输入张量。

## 接口

### 计算

```c
infiniStatus_t infiniopEqual(
    infiniopEqualDescriptor_t desc,
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
  已使用 `infiniopCreateEqualDescriptor()` 初始化的算子描述符。
- `workspace`:
  额外工作空间地址。
- `workspace_size`:
  `workspace` 的大小，单位：字节。
- `c`:
  计算结果地址，类型为布尔标量。
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
infiniStatus_t infiniopCreateEqualDescriptor(
    infiniopHandle_t handle,
    infiniopEqualDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t c,
    infiniopTensorDescriptor_t a,
    infiniopTensorDescriptor_t b
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数：</div>

- `handle`:
  硬件控柄。详见 [`InfiniopHandle_t`]。
- `desc_ptr`:
  `infiniopEqualDescriptor_t` 指针，指向将被初始化的算子描述符地址。
- `c` - { BOOL | () | (...) }:
  计算结果 `c` 的张量描述，必须是布尔标量。
- `a` - { dT | (d1,...,dn) | (...) }:
  输入张量 `a` 的张量描述，支持多向广播。
- `b` - { dT | (d1,...,dn) | (...) }:
  输入张量 `b` 的张量描述，支持多向广播。

参数限制：

- `dT`: 所有合法类型（`Float16`, `Float32`, `Float64`, `BFloat16`, `Int8`, `Int16`, `Int32`, `Int64`, `Uint8`, `Uint16`, `Uint32`, `Uint64`, `Bool`）。
- 输入张量 `a` 和 `b` 必须具有相同的形状和数据类型。
- 输出 `c` 必须是布尔标量（形状为空或所有维度为1）。
- 不支持原位计算。

<div style="background-color: lightblue; padding: 1px;"> 返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_BAD_PARAM`], [`INFINI_STATUS_BAD_TENSOR_SHAPE`], [`INFINI_STATUS_BAD_TENSOR_DTYPE`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`].

### 计算额外工作空间

```c
infiniStatus_t infiniopGetEqualWorkspaceSize(
    infiniopEqualDescriptor_t desc,
    size_t *size
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数：</div>

- `desc`:
  已使用 `infiniopCreateEqualDescriptor()` 初始化的算子描述符。
- `size`:
  额外空间大小的计算结果的写入地址。

<div style="background-color: lightblue; padding: 1px;"> 返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_NULL_POINTER`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`].

### 销毁算子描述符

```c
infiniStatus_t infiniopDestroyEqualDescriptor(
    infiniopEqualDescriptor_t desc
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
[`INFINI_STATUS_NULL_POINTER`]:/common/status/README.md#INFINI_STATUS_NULL_POINTER
[`INFINI_STATUS_INSUFFICIENT_WORKSPACE`]:/common/status/README.md#INFINI_STATUS_INSUFFICIENT_WORKSPACE
[`INFINI_STATUS_INTERNAL_ERROR`]:/common/status/README.md#INFINI_STATUS_INTERNAL_ERROR