# `infinicore::nn::Embedding`

嵌入层，将索引映射到密集向量。实现位于 `InfiniCore/src/infinicore/nn/embedding.cc`，头文件定义于 `InfiniCore/include/infinicore/nn/embedding.hpp`。

## 类定义

```cpp
namespace infinicore::nn {
    class Embedding : public Module {
    public:
        Embedding(size_t num_embeddings,
                  size_t embedding_dim,
                  std::optional<int64_t> padding_idx = std::nullopt,
                  const DataType &dtype = DataType::F32,
                  const Device &device = Device());
        
        Tensor forward(const Tensor &indices) const;
        std::string extra_repr() const;
        
        Tensor weight() const;
        size_t num_embeddings() const;
        size_t embedding_dim() const;
        std::optional<int64_t> padding_idx() const;
    };
}
```

## 构造函数参数

- `num_embeddings`：嵌入字典的大小（词汇表大小）。
- `embedding_dim`：每个嵌入向量的维度。
- `padding_idx`：可选，如果指定，该索引处的嵌入向量不会更新，也不参与梯度计算。
- `dtype`：嵌入权重的数据类型，默认为 `DataType::F32`。
- `device`：嵌入权重所在的设备。

## 主要方法

- `forward(indices)`：前向传播，根据索引查找对应的嵌入向量。
- `weight()`：获取嵌入权重张量。
- `num_embeddings()`：获取词汇表大小。
- `embedding_dim()`：获取嵌入维度。

## 输入输出

- 输入：索引张量，可以是任意形状 `(*)`，通常是 `[batch_size]` 或 `[batch_size, seq_len]`。
- 输出：嵌入向量张量，形状为 `(*, embedding_dim)`，其中 `*` 与输入形状匹配。

## 示例

```cpp
#include <infinicore.hpp>

using namespace infinicore;
using namespace infinicore::nn;

Device device(Device::Type::NVIDIA, 0);
context::setDevice(device);

// 创建嵌入层：10000 个词，300 维嵌入
Embedding embedding(10000, 300, std::nullopt, DataType::F16, device);

// 输入：形状 [batch_size, seq_len] = [2, 5]
Tensor indices = Tensor::empty({2, 5}, DataType::I64, device);
// ... 填充索引值 ...

// 输出：形状 [batch_size, seq_len, embedding_dim] = [2, 5, 300]
Tensor embeddings = embedding.forward(indices);

// 访问权重
Tensor weight = embedding.weight();
```

## 相关链接

- [`nn` 模块概览](../README.md)
- [`Module` 基类](../module/README.md)
