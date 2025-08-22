# `CrossEntropyLoss`

交叉熵损失（Cross Entropy Loss）算子，常用于分类任务的损失计算。该算子支持对 logits 与目标标签（target）进行交叉熵损失值的计算。

## 接口

### 计算

```c
infiniStatus_t infiniopCrossEntropyLoss(
    infiniopCrossEntropyLossDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *loss,
    const void *logits,
    const void *target,
    void *stream
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数： </div>

- `desc`：已通过 `infiniopCreateCrossEntropyLossDescriptor()` 初始化的交叉熵损失算子描述符。
- `workspace`：临时工作空间指针。
- `workspace_size`：工作空间字节数。
- `loss`：损失输出指针。
- `logits`：logits 输入张量指针。
- `target`：标签输入张量指针。
- `stream`：计算流/队列。

<div style="background-color: lightblue; padding: 1px;">  返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_BAD_PARAM`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`]

---

### 获取交叉熵损失临时工作空间需求

```c
infiniStatus_t infiniopGetCrossEntropyLossWorkspaceSize(
    infiniopCrossEntropyLossDescriptor_t desc,
    size_t *size
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数：</div>

- `desc`：已创建的交叉熵损失算子描述符。
- `size`：指向 `size_t` 的指针，返回所需的工作空间字节数。

<div style="background-color: lightblue; padding: 1px;"> 返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_BAD_PARAM`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`]

---

### 创建交叉熵损失算子描述符

```c
infiniStatus_t infiniopCreateCrossEntropyLossDescriptor(
    infiniopHandle_t handle,
    infiniopCrossEntropyLossDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t loss_desc,
    infiniopTensorDescriptor_t logits_desc,
    infiniopTensorDescriptor_t target_desc
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数：</div>

- `handle`: 硬件控柄。详情请见 [`InfiniopHandle_t`]
- `desc_ptr`: `infiniopCrossEntropyLossDescriptor_t` 指针，指向将被初始化的算子描述符地址。
- `loss_desc`：损失输出张量描述。
- `logits_desc`：logits 输入张量描述。
- `target_desc`：目标标签输入张量描述。

参数限制：

- **`dT`**:  (`Float16`, `Float32`, `Float64`) 之一。

<div style="background-color: lightblue; padding: 1px;"> 返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_BAD_PARAM`],  [`INFINI_STATUS_BAD_TENSOR_SHAPE`], [`INFINI_STATUS_BAD_TENSOR_DTYPE`], [`INFINI_STATUS_BAD_TENSOR_STRIDES`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`].

---

### 销毁交叉熵损失算子描述符

```c
infiniStatus_t infiniopDestroyCrossEntropyLossDescriptor(
    infiniopCrossEntropyLossDescriptor_t desc
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数： </div>

- `desc`
     : 待销毁的交叉熵损失算子描述符。

<div style="background-color: lightblue; padding: 1px;"> 返回值： </div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`].

<!-- 链接 -->
[`InfiniopHandle_t`]: /infiniop/handle/README.md

[`INFINI_STATUS_SUCCESS`]: /common/status/README.md#INFINI_STATUS_SUCCESS
[`INFINI_STATUS_BAD_PARAM`]: /common/status/README.md#INFINI_STATUS_BAD_PARAM
[`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`]: /common/status/README.md#INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED
[`INFINI_STATUS_BAD_TENSOR_SHAPE`]: /common/status/README.md#INFINI_STATUS_BAD_TENSOR_SHAPE
[`INFINI_STATUS_BAD_TENSOR_DTYPE`]: /common/status/README.md#INFINI_STATUS_BAD_TENSOR_DTYPE
[`INFINI_STATUS_BAD_TENSOR_STRIDES`]: /common/status/README.md#INFINI_STATUS_BAD_TENSOR_STRIDES