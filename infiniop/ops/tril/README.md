
# `Tril`

`Tril`算子返回输入2D张量`input`的对角分界线下三角部分，其他元素设置为 0。
参数 `diagonal` 控制的此操作的对角分界线位置。如果`diagonal`为 0，则保留主对角线及其下的所有元素。以此为基准，`diagonal`取正值对应向上偏移，负值对应向下偏移。


## 接口

### 计算

```c
infiniStatus_t infiniopTril(
    infiniopTrilDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void * output,
    const void * input,
    void *stream
);
```
<div style="background-color: lightblue; padding: 1px;"> 参数： </div>

- `desc`:
  已使用 `infiniopCreateTrilDescriptor()` 初始化的算子描述符。
- `workspace`:
  指向算子计算所需的额外工作空间；
- `workspace_size`:
  `workspace` 的大小，单位：字节；
- `output`:输出张量。张量限制见[创建算子描述](#创建算子描述)部分。
- `input`:输入张量。张量限制见[创建算子描述](#创建算子描述)部分。
 - `stream`: 计算流/队列。

<div style="background-color: lightblue; padding: 1px;">  返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_BAD_PARAM`], [`INFINI_STATUS_INSUFFICIENT_WORKSPACE`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`], [`INFINI_STATUS_INTERNAL_ERROR`]，[`INFINI_STATUS_BAD_TENSOR_DTYPE`].

---

### 创建算子描述

```c
infiniStatus_t infiniopCreateTrilDescriptor(
    infiniopHandle_t handle,
    infiniopTrilDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t input_desc,
    int diagonal
);
```
<div style="background-color: lightblue; padding: 1px;"> 参数：</div>

- `handle`:
  `infiniopHandle_t` 类型的硬件控柄。详情请看：[`InfiniopHandle_t`]。
- `desc_ptr`:
  `infiniopAddDescriptor_t` 指针，指向将被初始化的算子描述符地址；
- `output_desc` - { dT | (M, N) | (N, 1)}:
     算子计算参数 `output` 的张量描述。
- `input_desc` - { dT | (M, N) | (N, 1)}:
     算子计算参数 `input` 的张量描述。
- `diagonal` - int:
     设定对角分界线位置的参数。

参数限制：

- **`dT`**:  全部合法类型
- `output`与`input`均为2D全连续张量。

<div style="background-color: lightblue; padding: 1px;"> 返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_BAD_PARAM`], [`INFINI_STATUS_BAD_TENSOR_SHAPE`], [`INFINI_STATUS_BAD_TENSOR_DTYPE`], [`INFINI_STATUS_BAD_TENSOR_STRIDES`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`].
---

### 计算额外工作空间

```c
infiniStatus_t infiniopGetTrilWorkspaceSize(
    infiniopTrilDescriptor_t desc,
    size_t *size
);
```
<div style="background-color: lightblue; padding: 1px;"> 参数：</div>

- `desc`: 使用 `infiniopCreateTrilDescriptor()` 初始化的算子描述符。
- `size`:
  额外空间大小的计算结果的写入地址；

<div style="background-color: lightblue; padding: 1px;"> 返回值：</div>

 - [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_BAD_PARAM`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`].

---

### 销毁算子描述符

```c
infiniopStatus_t infiniopDestroyTrilDescriptor(
    infiniopTrilDescriptor_t desc
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
