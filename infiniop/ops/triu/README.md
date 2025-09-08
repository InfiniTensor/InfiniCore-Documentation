# `Triu`

`Triu`，即**上三角**算子。该算子返回输入张量的上三角部分，其余部分设置为零。其计算可被表述为：

$$ output[i][j] = \begin{cases} 
input[i][j] & \text{if } j \geq i + diagonal \\
0 & \text{otherwise}
\end{cases} $$

其中 `input` 为输入张量，`output` 为输出张量，`diagonal` 为对角线偏移量。

参考 `torch.triu` 实现，只需要支持2D连续张量。

## 接口

### 计算

```c
infiniStatus_t infiniopTriu(
    infiniopTriuDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *input,
    void *stream
);

infiniStatus_t infiniopTriuInplace(
    infiniopTriuDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *input_output,
    void *stream
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数： </div>

- `desc`:
  已使用 `infiniopCreateTriuDescriptor()` 初始化的算子描述符；
- `workspace`:
  指向算子计算所需的额外工作空间；
- `workspace_size`:
  `workspace` 的大小，单位：字节；
- `output`:
  输出张量。张量限制见[创建算子描述](#创建算子描述)部分；
- `input`:
  输入张量。张量限制见[创建算子描述](#创建算子描述)部分；
- `input_output`:
  输入输出张量（用于Inplace版本）。张量限制见[创建算子描述](#创建算子描述)部分；
- `stream`:
  计算流/队列；

<div style="background-color: lightblue; padding: 1px;"> 返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_BAD_PARAM`], [`INFINI_STATUS_INSUFFICIENT_WORKSPACE`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`], [`INFINI_STATUS_INTERNAL_ERROR`]，[`INFINI_STATUS_BAD_TENSOR_DTYPE`].

### 创建算子描述

```c
infiniStatus_t infiniopCreateTriuDescriptor(
    infiniopHandle_t handle,
    infiniopTriuDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t input_desc,
    int diagonal
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数：</div>

- `handle`:
  `infiniopHandle_t` 类型的硬件控柄。详情请看：[`InfiniopHandle_t`]。
- `desc_ptr`:
  `infiniopTriuDescriptor_t` 指针，指向将被初始化的算子描述符地址；
- `output_desc` - { dT | (M, N) | (...) }:
  算子计算参数 `output` 的张量描述，支持原位计算。
- `input_desc` - { dT | (M, N) | (...) }:
  算子计算参数 `input` 的张量描述。
- `diagonal`:
  对角线偏移量。默认为0（主对角线）。

参数限制：

- `dT`: 所有合法类型。
- 只支持2D连续张量。
- 支持原位计算，即计算时 `output` 可以和 `input` 指向同一地址。
- `input` 和 `output` 张量必须具有相同的形状 `(M, N)`。

<div style="background-color: lightblue; padding: 1px;"> 返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_BAD_PARAM`], [`INFINI_STATUS_BAD_TENSOR_SHAPE`], [`INFINI_STATUS_BAD_TENSOR_DTYPE`], [`INFINI_STATUS_BAD_TENSOR_STRIDES`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`].

### 计算额外工作空间

```c
infiniStatus_t infiniopGetTriuWorkspaceSize(
    infiniopTriuDescriptor_t desc,
    size_t *size
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数：</div>

- `desc`:
  已使用 `infiniopCreateTriuDescriptor()` 初始化的算子描述符；
- `size`:
  额外空间大小的计算结果的写入地址；

<div style="background-color: lightblue; padding: 1px;"> 返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_NULL_POINTER`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`].

### 销毁算子描述符

```c
infiniStatus_t infiniopDestroyTriuDescriptor(
    infiniopTriuDescriptor_t desc
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