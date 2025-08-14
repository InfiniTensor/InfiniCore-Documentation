
# `Index Copy Inplace`

对于输入张量`input`、`index`和输出张量`output`，`Index Copy Inplace` 将 `input` 中的元素按 `index` 提供的针对`dim`维度的顺序复制到 `output` 张量中。

例如，如果 `dim` 等于 0 且 `index`[i] 等于 j，则 `input` 的第 i 行将被复制到 `output` 的第 j 行。

## 接口

### 计算

```c
infiniStatus_t infiniopIndexCopyInplace(
    infiniopIndexCopyInplaceDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void * output,
    const void * input,
    const void * index,
    void *stream
);
```
<div style="background-color: lightblue; padding: 1px;"> 参数： </div>

- `desc`:
  已使用 `infiniopCreateIndexCopyInplaceDescriptor()` 初始化的算子描述符。
- `workspace`:
  指向算子计算所需的额外工作空间；
- `workspace_size`:
  `workspace` 的大小，单位：字节；
- `output`:输出张量。张量限制见[创建算子描述](#创建算子描述)部分。
- `input`:输入张量。张量限制见[创建算子描述](#创建算子描述)部分。
- `index`:输入张量。张量限制见[创建算子描述](#创建算子描述)部分。
 - `stream`: 计算流/队列。

<div style="background-color: lightblue; padding: 1px;">  返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_BAD_PARAM`], [`INFINI_STATUS_INSUFFICIENT_WORKSPACE`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`], [`INFINI_STATUS_INTERNAL_ERROR`]，[`INFINI_STATUS_BAD_TENSOR_DTYPE`].

---

### 创建算子描述

```c
infiniStatus_t infiniopCreateIndexCopyInplaceDescriptor(
    infiniopHandle_t handle,
    infiniopIndexCopyInplaceDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t input_desc,
    infiniopTensorDescriptor_t index_desc,
    size_t dim
);
```
<div style="background-color: lightblue; padding: 1px;"> 参数：</div>

- `handle`:
  `infiniopHandle_t` 类型的硬件控柄。详情请看：[`InfiniopHandle_t`]。
- `desc_ptr`:
  `infiniopAddDescriptor_t` 指针，指向将被初始化的算子描述符地址；
- `output_desc` - { dT | (d1, ..., $\rm{d_{o,dim}}$,..., dn) | ($...$) }:
     算子计算参数 `output` 的张量描述。
- `input_desc` - { dT | (d1, ..., $\rm{d_{i,dim}}$,..., dn) | ($...$) }:
     算子计算参数 `input` 的张量描述。
- `index_desc` - { int | ($\rm{d_{i,dim}}$) | ($...$) }:
     算子计算参数 `index` 的张量描述，每一个元素值i应满足 0 $\leq$ i $\lt$ $\rm{d_{o,dim}}$。
- `dim` - int:
    算子针对的维度， 满足 0 $\leq$ dim $\lt$ n。

参数限制：

- **`dT`**:  所有合法类型
- `output`与`input`除`dim`外全部维度长度相等。
- 如果`input`中多组指向`output`同一位置，会出现不确定性。故测试时，通过规定`index`的不可重复，避免此情况发生。

<div style="background-color: lightblue; padding: 1px;"> 返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_BAD_PARAM`], [`INFINI_STATUS_BAD_TENSOR_SHAPE`], [`INFINI_STATUS_BAD_TENSOR_DTYPE`], [`INFINI_STATUS_BAD_TENSOR_STRIDES`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`].
---

### 计算额外工作空间

```c
infiniStatus_t infiniopGetIndexCopyInplaceWorkspaceSize(
    infiniopIndexCopyInplaceDescriptor_t desc,
    size_t *size
);
```
<div style="background-color: lightblue; padding: 1px;"> 参数：</div>

- `desc`: 使用 `infiniopCreateIndexCopyInplaceDescriptor()` 初始化的算子描述符。
- `size`:
  额外空间大小的计算结果的写入地址；

<div style="background-color: lightblue; padding: 1px;"> 返回值：</div>

 - [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_BAD_PARAM`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`].

---

### 销毁算子描述符

```c
infiniopStatus_t infiniopDestroyIndexCopyInplaceDescriptor(
    infiniopIndexCopyInplaceDescriptor_t desc
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
