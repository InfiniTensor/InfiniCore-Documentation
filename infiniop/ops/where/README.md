# `Where`

`Where`，即**条件选择**算子。对每个元素位置，根据布尔条件张量 `condition`，从 `a` 或 `b` 中选择对应元素写入输出张量 `c`。本算子等价于三目运算：`c = condition ? a : b`。

对输入张量 `a`、`b`，布尔张量 `condition`，输出张量 `c`，逐元素计算如下：

$$
c_i =
\begin{cases}
a_i & \text{if } condition_i = \text{true} \\
b_i & \text{if } condition_i = \text{false}
\end{cases}
$$

示例：
若 `a = [-1, 10, 3]`，`b = [0, 20, 4]`，`condition = [true, false, true]`，则输出 `c = [-1, 20, 3]`。
当存在形状不一致时，`a`、`b` 与 `condition` 需可按多向广播规则与 `c` 对齐。

## 接口

### 计算

```c
infiniStatus_t infiniopWhere(
    infiniopWhereDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *c,
    const void *condition,
    const void *a,
    const void *b,
    void *stream
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数： </div>

- `desc`：
  已使用 `infiniopCreateWhereDescriptor()` 初始化的算子描述符。
- `workspace`：
  算子执行所需的工作空间指针。
- `workspace_size`：
  工作空间大小（字节）。
- `c`：
  输出张量。张量限制见[创建算子描述](#创建算子描述)部分；
- `a`：
  输入张量；当 `condition` 为 `true` 时取用。张量限制见[创建算子描述](#创建算子描述)部分；
- `b`：
  输入张量；当 `condition` 为 `false` 时取用。张量限制见[创建算子描述](#创建算子描述)部分；
- `condition`：
  布尔条件张量（`Bool`），决定在各元素位置从 `a` 或 `b` 取值。张量限制见[创建算子描述](#创建算子描述)部分；
- `stream`：
  计算流/队列。

<div style="background-color: lightblue; padding: 1px;"> 返回值： </div>

- [`INFINI_STATUS_SUCCESS`],[`INFINI_STATUS_BAD_TENSOR_DTYPE`],[`INFINI_STATUS_BAD_TENSOR_SHAPE`],[`INFINI_STATUS_INSUFFICIENT_WORKSPACE`]
,[`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`].

### 创建算子描述符

```c
infiniStatus_t infiniopCreateWhereDescriptor(
    infiniopHandle_t handle,
    infiniopWhereDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t c,
    infiniopTensorDescriptor_t condition,
    infiniopTensorDescriptor_t a,
    infiniopTensorDescriptor_t b
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数： </div>

- `handle`：
  `infiniopHandle_t` 类型的硬件控柄。详见：[`InfiniopHandle_t`]
- `desc_ptr`：
  `infiniopWhereDescriptor_t` 指针，指向将被初始化的算子描述符地址。
- `c` - {dT | (d1, ..., dn) | (...)}：
  输出张量描述。其形状应为 `a`、`b`、`condition` 的广播结果。
- `a` - {dT | (d1, ..., dn) | (...)}：
  输入张量描述。
- `b` - {dT | (d1, ..., dn) | (...)}：
  输入张量描述。
- `condition` - {Bool | (d1, ..., dn) | (...)}：
  布尔条件张量描述。

参数限制：

- `dT`: (`Float16`, `Float32`, `Float64`, `BFloat16`，`Bool`,`Int8`, `Int16`, `Int32`, `Int64`,`Uint8`, `Uint16`, `Uint32`, `Uint64`) 之一。
- `a`、`b` 与 `condition` 必须与 `c` 可广播；`c` 的形状为三者广播后的结果。发生多向广播时需通过步长设置完成映射关系。
- 支持原位计算，即 `c` 可以与 `a` 或 `b` 指向同一地址。
- 计算输出参数 `c` 不能进行广播（`c` 的步长不能包含 0）。

<div style="background-color: lightblue; padding: 1px;"> 返回值： </div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_BAD_PARAM`], [`INFINI_STATUS_BAD_TENSOR_SHAPE`], [`INFINI_STATUS_BAD_TENSOR_DTYPE`], [`INFINI_STATUS_BAD_TENSOR_STRIDES`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`].

### 计算额外工作空间

```c
infiniStatus_t infiniopGetWhereWorkspaceSize(
    infiniopWhereDescriptor_t desc,
    size_t *size
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数： </div>

- `desc`：
  输入。已初始化的算子描述符。
- `workspace_size`：
  输出。算子执行所需的工作空间大小（字节）。

<div style="background-color: lightblue; padding: 1px;"> 返回值： </div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_NULL_POINTER`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`].

### 销毁算子描述符

```c
infiniStatus_t infiniopDestroyWhereDescriptor(
    infiniopWhereDescriptor_t desc
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数： </div>

- `desc`：
  输入。待销毁的算子描述符。

<div style="background-color: lightblue; padding: 1px;"> 返回值： </div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`].

## 已知问题

无

<!-- 链接 -->
[`InfiniopHandle_t`]: /infiniop/handle/README.md

[`INFINI_STATUS_SUCCESS`]: /common/status/README.md#INFINI_STATUS_SUCCESS
[`INFINI_STATUS_BAD_PARAM`]: /common/status/README.md#INFINI_STATUS_BAD_PARAM
[`INFINI_STATUS_BAD_TENSOR_SHAPE`]: /common/status/README.md#INFINI_STATUS_BAD_TENSOR_SHAPE
[`INFINI_STATUS_BAD_TENSOR_DTYPE`]: /common/status/README.md#INFINI_STATUS_BAD_TENSOR_DTYPE
[`INFINI_STATUS_BAD_TENSOR_STRIDES`]: /common/status/README.md#INFINI_STATUS_BAD_TENSOR_STRIDES
[`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`]: /common/status/README.md#INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED
[`INFINI_STATUS_INTERNAL_ERROR`]: /common/status/README.md#INFINI_STATUS_INTERNAL_ERROR
[`INFINI_STATUS_INSUFFICIENT_WORKSPACE`]: /common/status/README.md#INFINI_STATUS_INSUFFICIENT_WORKSPACE
