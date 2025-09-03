# `FlashAttentionBackward`

`FlashAttentionBackward` 是算子 `FlashAttention` 的反向传播。

与正向传播相同，先进行分块。 $K$ 和 $V$ 在外循环中逐块加载，而 $Q$ 在内循环中逐块加载。

1. 外循环中，每次循环需要先初始化当前块的 $\mathbf{d}\mathbf{K}_j$ 和 $\mathbf{d}\mathbf{V}_j$ 为 0；
2. 内循环中，按以下顺序计算
    1. $\mathbf{S}_i^{(j)}=\mathbf{Q}_i\mathbf{K}_j^T\in\mathbb{R}^{B_r\times B_c}$
    2. $\mathbf{P}_i^{(j)}=\exp(\mathbf{S} _{ij}-L_i)\in\mathbb{R}^{B_r\times B_c}$
    3. $\mathbf{d}\mathbf{V}_j\leftarrow\mathbf{d}\mathbf{V}_j+(\mathbf{P}_i^{(j)})^\top\mathbf{d}\mathbf{O}_i\in\mathbb{R}^{B_c\times d}$
    4. $\mathbf{dP}_i^{(j)}=\mathbf{dO}_i\mathbf{V}_j^\top\in\mathbb{R}^{B_r\times B_c}$
    5. $\mathbf{dQ}_i\leftarrow\mathbf{dQ}_i+\mathbf{dS}_i^{(j)}\mathbf{K}_j$
    6. $\mathbf{dK}_j\leftarrow\mathbf{dK}_j+\mathbf{dS}_i^{(j)\top}\mathbf{Q}_i\in\mathbb{R}^{B_c\times d}$

## 接口

### 计算

```c
infiniStatus_t infiniopFlashAttentionBackward(
    infiniopFlashAttentionBackwardDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *grad_q,
    void *grad_k,
    void *grad_v,
    const void *q,
    const void *k,
    const void *v,
    const void *grad_out,
    const void *mask,
    void *stream
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数： </div>

- `desc`:
  已使用 `infiniopCreateFlashAttentionBackwardDescriptor()` 初始化的算子描述符；
- `workspace`:
  指向算子计算所需的额外工作空间；
- `workspace_size`:
  `workspace` 的大小，单位：字节；
- `grad_q`:
  查询（Query）梯度计算结果地址。张量限制见[创建算子描述](#创建算子描述)部分。
- `grad_k`:
  键（Key）梯度计算结果地址。张量限制见[创建算子描述](#创建算子描述)部分。
- `grad_v`:
  值（Value）梯度计算结果地址。张量限制见[创建算子描述](#创建算子描述)部分。
- `q`:
  查询（Query）张量数据指针。张量限制见[创建算子描述](#创建算子描述)部分。
- `k`:
  键（Key）张量数据指针。张量限制见[创建算子描述](#创建算子描述)部分。
- `v`:
  值（Value）张量数据指针。张量限制见[创建算子描述](#创建算子描述)部分。
- `grad_out`
  注意力梯度张量数据指针。张量限制见[创建算子描述](#创建算子描述)部分。
- `mask`:
  注意力掩码的数据指针，可选参数。取值为 `0` 时表示保留对应位置的元素（参与计算）；取值为 `-inf` （负无穷）时表示屏蔽对应位置的元素（即跳过，不参与计算）。张量限制见[创建算子描述](#创建算子描述)部分。
- `stream`:
  计算流/队列。

<div style="background-color: lightblue; padding: 1px;">  返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_NULL_POINTER`], [`INFINI_STATUS_INSUFFICIENT_WORKSPACE`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`], [`INFINI_STATUS_INTERNAL_ERROR`].

### 创建算子描述

```c
infiniStatus_t infiniopCreateFlashAttentionBackwardDescriptor(
    infiniopHandle_t handle,
    infiniopFlashAttentionBackwardDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t grad_q_desc,
    infiniopTensorDescriptor_t grad_k_desc,
    infiniopTensorDescriptor_t grad_v_desc,
    infiniopTensorDescriptor_t q_desc,
    infiniopTensorDescriptor_t k_desc,
    infiniopTensorDescriptor_t v_desc,
    infiniopTensorDescriptor_t grad_out_desc,
    infiniopTensorDescriptor_t mask_desc,
    infiniopAttentionMaskType_t mask_type
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数：</div>

- `handle`:
  `infiniopHandle_t`类型的硬件控柄。详见 [`InfiniopHandle_t`]。
- `desc_ptr`:
  `infiniopFlashAttentionBackwardDescriptor_t` 指针，指向将被初始化的算子描述符地址。
- `grad_q_desc` - { dT | ((batch_size,) seq_len_q, num_heads_q, head_dim) | ($\ldots, 1$)}:
  算子计算参数 `grad_q` 的张量描述，四维或者三维，最后一维连续。
- `grad_k_desc` - { dT | ((batch_size,) seq_len_kv, num_heads_kv, head_dim) | ($\ldots, 1$)}:
  算子计算参数 `grad_k` 的张量描述，四维或者三维，最后一维连续。
- `grad_v_desc` - { dT | ((batch_size,) seq_len_kv, num_heads_kv, head_dim) | ($\ldots, 1$)}:
  算子计算参数 `grad_v` 的张量描述，四维或者三维，最后一维连续。
- `q_desc` - { dT | ((batch_size,) seq_len_q, num_heads_q, head_dim) | ($\ldots, 1$)}:
  算子计算参数 `q` 的张量描述，形状与 `grad_q_desc` 一致，最后一维连续。
- `k_desc` - { dT | ((batch_size,) seq_len_kv, num_heads_kv, head_dim) | ($\ldots, 1$)}:
  算子计算参数 `k` 的张量描述，形状与 `grad_k_desc` 一致，最后一维连续。
- `v_desc` - { dT | ((batch_size,) seq_len_kv, num_heads_kv, head_dim) | ($\ldots, 1$)}:
  算子计算参数 `v` 的张量描述，形状与 `grad_v_desc` 一致，最后一维连续。
- `grad_out` - { dT | ((batch_size,) seq_len_q, num_heads_q, head_dim) | ($\ldots, 1$)}:
  算子计算参数 `grad_out` 的张量描述，形状与 `grad_q_desc` 一致，最后一维连续。
- `mask_desc` - { dM | (seq_len_q, seq_len_kv) | (~)}:
  算子计算参数 `mask` 的张量描述，当 `mask_type=INFINIOP_ATTENTION_MASK_TYPE_FULL` 时，`mask` 不可为空，其余情况 `mask` 可为`nullptr`。
- `mask_type` - `infiniopAttentionMaskType_t`:
  注意力类型参数，有三种类型可选，详细见参数限制。

参数限制：

- `dT`: `Float16`, `Float32` 或 `BFloat16`。
- `dM`: `Flaot32`。
- `seq_len_q` 与 `seq_len_kv` 可以不同。
- `num_heads_q` 与 `num_heads_kv` 可以不同，但需满足前者是后者的整数倍（非0整数）。
  - 当 $N_q/N_{kv}=1$ 时，即为 MQA (multi-query attention)
  - 当 $N_q/N_{kv}>1$ 时，即为 GQA (grouped-query attention)
- `mask_type` 的三种类型：
  - `INFINIOP_ATTENTION_MASK_TYPE_NONE=0`: 不使用注意力掩码，忽略 `mask` 取值；
  - `INFINIOP_ATTENTION_MASK_TYPE_FULL=1`: 使用完整 mask 矩阵，此时 `mask` 不可为空；
  - `INFINIOP_ATTENTION_MASK_TYPE_CAUSAL=2`: 使用标准因果掩码，对应以左上顶点划分的下三角场景，忽略 `mask` 取值；

<div style="background-color: lightblue; padding: 1px;"> 返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_BAD_PARAM`],  [`INFINI_STATUS_BAD_TENSOR_SHAPE`], [`INFINI_STATUS_BAD_TENSOR_DTYPE`], [`INFINI_STATUS_BAD_TENSOR_STRIDES`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`].

### 计算额外工作空间

```c
infiniStatus_t infiniopGetFlashAttentionBackwardWorkspaceSize(
    infiniopFlashAttentionBackwardDescriptor_t desc,
    size_t *size
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数：</div>

- `desc`:
  已使用 `infiniopCreateFlashAttentionBackwardDescriptor()` 初始化的算子描述符；
- `size`:
  额外空间大小的计算结果的写入地址；

<div style="background-color: lightblue; padding: 1px;"> 返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_BAD_PARAM`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`].

### 销毁算子描述符

```c
infiniStatus_t infiniopDestoryFlashAttentionBackwardDescriptor(
    infiniopFlashAttentionBackwardDescriptor_t desc
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数： </div>

- `desc`:
  输入。 待销毁的算子描述符；

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
[`INFINI_STATUS_NULL_POINTER`]:/common/status/README.md#INFINI_STATUS_NULL_POINTER
[`INFINI_STATUS_INSUFFICIENT_WORKSPACE`]:/common/status/README.md#INFINI_STATUS_INSUFFICIENT_WORKSPACE
[`INFINI_STATUS_INTERNAL_ERROR`]:/common/status/README.md#INFINI_STATUS_INTERNAL_ERROR
