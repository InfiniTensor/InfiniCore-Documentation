
# LeakyReLU

LeakyReLU, 即 **非线性激活函数**算子，为单目逐元素算子。其计算可被表述为：

$$ 
output = \text{LeakyReLU}({input}) =
\begin{cases}
{input}, & \text{if } {input} \geq 0 \\
{negative\\_slope}* {input}, & \text{if } {input} < 0
\end{cases} 
$$

其中 `input` 和  为输入，`output` 为输出, `negative_slope`为构建函数时的常数。

## 接口

### 计算

```c
infiniStatus_t infiniopLeakyRelu(
    infiniopLeakyReluDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *input,
    float negative_slope,
    void *stream
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数： </div>

- `desc`:
  已使用 `infiniopCreateAddDescriptor()` 初始化的算子描述符；
  
- `workspace`:
  指向算子计算所需的额外工作空间；
  
- `workspace_size`:
  `workspace` 的大小，单位：字节；
  
- `output`:
  输出张量。张量限制见[创建算子描述](#创建算子描述)部分；
  
- `input`:
  输入张量。张量限制见[创建算子描述](#创建算子描述)部分；
  
- `negative_slope`
  
  常量，构建函数时设置
  
- `stream`:
  计算流/队列；

<div style="background-color: lightblue; padding: 1px;"> 返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_INSUFFICIENT_WORKSPACE`], [`INFINI_STATUS_BAD_TENSOR_DTYPE`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`],  [`INFINI_STATUS_BAD_PARAM`], [`INFINI_STATUS_BAD_TENSOR_STRIDES`], [`INFINI_STATUS_BAD_TENSOR_SHAPE`]. 

### 创建算子描述

```c
infiniStatus_t infiniopCreateLeakyReluDescriptor(
    infiniopHandle_t handle,
    infiniopLeakyReluDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t output,
    infiniopTensorDescriptor_t input
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数：</div>

- `handle`:
  `infiniopHandle_t` 类型的硬件控柄。详情请看：[`InfiniopHandle_t`]。
- `desc_ptr`:
  `infiniopAddDescriptor_t` 指针，指向将被初始化的算子描述符地址；
- output_desc` - { dT | (d1,...,dn) | (...) }:
  算子计算参数 output` 的张量描述，支持原位计算。
- input_desc` - { dT | (d1,...,dn) | (...) }:
  算子计算参数 input` 的张量描述，支持原位计算。
  
  

参数限制：

- `dT`:  (`Float16`, `Float32`, `BFloat16`) 之一。
- 输入 `input `  的形状需与 `output` 相同。
- 支持原位计算，即计算时`input`  可以和 `output`  指向同一地址。

<div style="background-color: lightblue; padding: 1px;"> 返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_BAD_PARAM`], [`INFINI_STATUS_BAD_TENSOR_SHAPE`], [`INFINI_STATUS_BAD_TENSOR_DTYPE`], [`INFINI_STATUS_BAD_TENSOR_STRIDES`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`].

### 计算额外工作空间

```c
infiniStatus_t infiniopGetLeakyReluWorkspaceSize(
    infiniopLeakyReluDescriptor_t desc,
    size_t *size
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数：</div>

- `desc`:
  已使用 `infiniopCreateAddDescriptor()` 初始化的算子描述符；
- `size`:
  额外空间大小的计算结果的写入地址；

<div style="background-color: lightblue; padding: 1px;"> 返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_NULL_POINTER`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`].

### 销毁算子描述符

```c
infiniStatus_t infiniopDestroyLeakyReluDescriptor(
    infiniopLeakyReluDescriptor_t desc
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

