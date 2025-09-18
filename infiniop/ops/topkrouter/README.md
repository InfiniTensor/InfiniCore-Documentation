# `Topkrouter`

Deepseek的MoE中用到的`Topkrouter`算子，用于为每个token选择前topk个路由专家。



## 接口

### 计算

```c
infiniStatus_t infiniopTopkrouter(
    infiniopTopkrouterDescriptor_t desc, 
    void *workspace, 
    size_t workspace_size,
    void *values, 
    void *indices, 
    const void *x, 
    const void *correction_bias, 
    const float routed_scaling_factor, 
    const size_t topk, 
    void *stream
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数： </div>

- `desc`:
  使用 `infiniopCreateTopkrouterDescriptor()` 初始化的算子描述符。
- `workspace`:
  算子计算所需的额外工作空间。
- `workspace_size`:
  `workspace` 的大小，单位：字节。
- `values`:
  输出数据地址。表示每个token的前topk个路由专家的权重数据的地址，数据类型为Float32指针，形状为二维，(ntoken, topk)。ntoken是输入token的数量。
- `indices`:
  输出数据地址。表示每个token的前topk个路由专家的索引数据的地址，数据类型为Int32指针，形状为二维，(ntoken, topk)。
- `x`:
  输入数据地址。张量限制见[创建算子描述](#创建算子描述)部分中的 x_des。
- `correction_bias`:
  输入数据地址，表示偏置权重。张量限制见[创建算子描述](#创建算子描述)部分中的 correction_bias_desc。
- `routed_scaling_factor`:
  路由权重的缩放因子。
- `topk`:
  计算前topk个权重和索引。
- `stream`:
  计算流/队列。



<div style="background-color: lightblue; padding: 1px;"> 返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_BAD_PARAM`], [`INFINI_STATUS_BAD_TENSOR_DTYPE`],[`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`].



### 创建算子描述

```c
infiniStatus_t infiniopCreateTopkrouterDescriptor(
    infiniopHandle_t handle,
    infiniopTopkrouterDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t correction_bias_desc);
```
<div style="background-color: lightblue; padding: 1px;"> 参数：</div>

- `handle`:
  硬件控柄。详情请看：[`InfiniopHandle_t`]。
- `desc_ptr`:
  存放将被初始化的算子描述符地址。
- `x_desc` : 路由专家的gate数据，形状为二维，(ntoken, num_experts)。ntoken是输入token的数量，num_experts是路由专家的数量。
- `correction_bias_desc` : 每个专家的偏置权重参数，形状为一维，(num_experts，)。

参数限制：

- `x_desc`: 数据类型为(`Float16`, `Float32`, `BFloat16`) 之一； 第二个维度的路由专家数量num_experts大小固定为256。
- `correction_bias_desc`: 数据类型为`Float32`。

<div style="background-color: lightblue; padding: 1px;"> 返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_BAD_PARAM`],  [`INFINI_STATUS_BAD_TENSOR_DTYPE`],  [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`].



### 计算额外工作空间

```c
infiniStatus_t infiniopGetTopkrouterWorkspaceSize(
    infiniopTopkrouterDescriptor_t desc,
    size_t *size
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数：</div>

- `desc`:
  使用 `infiniopCreateTopkrouterDescriptor()` 初始化的算子描述符。
- `size`:
  存放空间大小的计算结果的地址。

<div style="background-color: lightblue; padding: 1px;"> 返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`].

### 销毁算子描述符

```c
infiniStatus_t infiniopDestroyTopkrouterDescriptor(
    infiniopTopkrouterDescriptor_t desc
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数： </div>

- `desc`:
  待销毁的算子描述符。

<div style="background-color: lightblue; padding: 1px;"> 返回值： </div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`].

<!-- 链接 -->
[`InfiniopHandle_t`]: /infiniop/handle/README.md

[`INFINI_STATUS_SUCCESS`]: /common/status/README.md#INFINI_STATUS_SUCCESS
[`INFINI_STATUS_BAD_PARAM`]: /common/status/README.md#INFINI_STATUS_BAD_PARAM
[`INFINI_STATUS_INSUFFICIENT_WORKSPACE`]: /common/status/README.md#INFINI_STATUS_INSUFFICIENT_WORKSPACE
[`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`]: /common/status/README.md#INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED
[`INFINI_STATUS_INTERNAL_ERROR`]: /common/status/README.md#INFINI_STATUS_INTERNAL_ERROR
[`INFINI_STATUS_BAD_TENSOR_SHAPE`]: /common/status/README.md#INFINI_STATUS_BAD_TENSOR_SHAPE
[`INFINI_STATUS_BAD_TENSOR_DTYPE`]: /common/status/README.md#INFINI_STATUS_BAD_TENSOR_DTYPE
[`INFINI_STATUS_BAD_TENSOR_STRIDES`]: /common/status/README.md#INFINI_STATUS_BAD_TENSOR_STRIDES
