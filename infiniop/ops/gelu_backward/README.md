# GeLU Backward 算子

GeLU Backward 算子用于计算 GeLU 激活函数的反向传播梯度。该算子参考 `torch.gelu` 的反向传播行为。

## 数学定义

给定输入张量 $x$ 和输出梯度 $\frac{\partial L}{\partial y}$，GeLU Backward 算子的计算公式为：

$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \cdot \frac{\partial \text{GeLU}(x)}{\partial x}$$

其中 GeLU 函数的导数为：

$$\frac{\partial \text{GeLU}(x)}{\partial x} = \Phi(x) + x \cdot \phi(x)$$

这里 $\Phi(x)$ 是标准正态分布的累积分布函数，$\phi(x) = \frac{1}{\sqrt{2\pi}}e^{-\frac{x^2}{2}}$ 是标准正态分布的概率密度函数。

近似计算的导数公式为：

$$\frac{\partial \text{GeLU}(x)}{\partial x} \approx 0.5 \cdot \tanh\left(\sqrt{\frac{2}{\pi}} \cdot (x + 0.044715 \cdot x^3)\right) + \frac{0.5 \cdot x \cdot \text{sech}^2\left(\sqrt{\frac{2}{\pi}} \cdot (x + 0.044715 \cdot x^3)\right) \cdot \sqrt{\frac{2}{\pi}} \cdot (1 + 0.134145 \cdot x^2)}{1}$$

## 接口

### 计算

```c
infiniStatus_t infiniopGeluBackward(
    infiniopGeluBackwardDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *grad_input,
    const void *input,
    const void *grad_output,
    void *stream
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数：</div>

- `desc`:
  已使用 `infiniopCreateGeluBackwardDescriptor()` 初始化的算子描述符。
- `workspace`:
  额外工作空间地址。
- `workspace_size`:
  `workspace` 的大小，单位：字节。
- `grad_input`:
  输入梯度的计算结果地址，支持原位计算。
- `input`:
  前向传播的输入张量数据指针。
- `grad_output`:
  输出梯度张量数据指针。
- `stream`:
  计算流/队列。

<div style="background-color: lightblue; padding: 1px;"> 返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_NULL_POINTER`], [`INFINI_STATUS_INSUFFICIENT_WORKSPACE`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`], [`INFINI_STATUS_INTERNAL_ERROR`].

### 创建算子描述

```c
infiniStatus_t infiniopCreateGeluBackwardDescriptor(
    infiniopHandle_t handle,
    infiniopGeluBackwardDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t grad_input,
    infiniopTensorDescriptor_t input,
    infiniopTensorDescriptor_t grad_output
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数：</div>

- `handle`:
  硬件控柄。详见 [`InfiniopHandle_t`]。
- `desc_ptr`:
  `infiniopGeluBackwardDescriptor_t` 指针，指向将被初始化的算子描述符地址。
- `grad_input` - { FP16 | FP32 | BF16 | (d1,...,dn) | (...) }:
  输入梯度张量描述，支持原位计算。
- `input` - { FP16 | FP32 | BF16 | (d1,...,dn) | (...) }:
  前向传播输入张量描述。
- `grad_output` - { FP16 | FP32 | BF16 | (d1,...,dn) | (...) }:
  输出梯度张量描述，支持原位计算。

参数限制：

- 所有张量的数据类型必须相同，支持 `FP16`、`FP32`、`BF16`。
- 所有张量的形状必须相同。
- 支持原位计算，即计算时 `grad_input` 可以和 `grad_output` 指向同一地址。

<div style="background-color: lightblue; padding: 1px;"> 返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_BAD_PARAM`], [`INFINI_STATUS_BAD_TENSOR_SHAPE`], [`INFINI_STATUS_BAD_TENSOR_DTYPE`], [`INFINI_STATUS_BAD_TENSOR_STRIDES`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`].

### 计算额外工作空间

```c
infiniStatus_t infiniopGetGeluBackwardWorkspaceSize(
    infiniopGeluBackwardDescriptor_t desc,
    size_t *size
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数：</div>

- `desc`:
  已使用 `infiniopCreateGeluBackwardDescriptor()` 初始化的算子描述符。
- `size`:
  额外空间大小的计算结果的写入地址。

<div style="background-color: lightblue; padding: 1px;"> 返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_NULL_POINTER`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`].

### 销毁算子描述符

```c
infiniStatus_t infiniopDestroyGeluBackwardDescriptor(
    infiniopGeluBackwardDescriptor_t desc
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