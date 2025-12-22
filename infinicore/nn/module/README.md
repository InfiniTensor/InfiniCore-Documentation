# `infinicore::nn::Module`

模块基类，提供参数管理、状态字典、子模块注册等功能。实现位于 `InfiniCore/src/infinicore/nn/module.cc`，头文件定义于 `InfiniCore/include/infinicore/nn/module.hpp`。

## 类定义

```cpp
namespace infinicore::nn {
    class Module {
    public:
        Module();
        
        const std::unordered_map<std::string, Parameter> &state_dict() const;
        void load_state_dict(const std::unordered_map<std::string, Tensor> &state_dict);
        void load_parameter(const std::string &name, const Tensor &param);
        void load_parameter_(const std::string &name, const Tensor &param);
        void load_parameter_from_blob(const std::string &name, const void *data);
        
    protected:
        Tensor register_parameter(const std::string &name, Parameter param);
        Tensor register_buffer(const std::string &name, Parameter buffer);
        
        template <typename M>
        std::shared_ptr<M> add_module(const std::string &name, std::shared_ptr<M> submodule);
        
        template <typename M, typename... Args>
        std::shared_ptr<M> register_module(const std::string &name, Args &&...args);
        
        template <typename M, typename... Args>
        std::vector<std::shared_ptr<M>> register_modules(size_t count, const std::string &name, Args &&...args);
    };
}
```

## 主要方法

### 参数管理

- `state_dict()`：获取所有参数的字典。
- `load_state_dict(state_dict)`：加载状态字典。
- `load_parameter(name, param)`：加载单个参数（创建新张量）。
- `load_parameter_(name, param)`：加载单个参数（原地更新）。
- `load_parameter_from_blob(name, data)`：从内存块加载参数。

### 参数注册

- `register_parameter(name, param)`：注册可训练参数。
- `register_buffer(name, buffer)`：注册缓冲区（不可训练）。

### 子模块管理

- `add_module(name, submodule)`：添加已有子模块。
- `register_module(name, ...)`：创建并注册新子模块。
- `register_modules(count, name, ...)`：创建并注册多个同类型子模块。

## 便捷宏

为了方便模块定义，提供了以下宏：

- `INFINICORE_NN_MODULE(ModuleType, name)`：声明子模块成员变量。
- `INFINICORE_NN_MODULE_INIT(name, ...)`：在构造函数中初始化子模块。
- `INFINICORE_NN_MODULE_VEC(ModuleType, name)`：声明子模块向量。
- `INFINICORE_NN_MODULE_VEC_INIT(name, count, ModuleType, ...)`：初始化子模块向量。
- `INFINICORE_NN_PARAMETER(name)`：声明参数成员变量。
- `INFINICORE_NN_PARAMETER_INIT(name, args)`：初始化参数。
- `INFINICORE_NN_BUFFER(name)`：声明缓冲区成员变量。
- `INFINICORE_NN_BUFFER_INIT(name, args)`：初始化缓冲区。

## 示例

```cpp
#include <infinicore.hpp>

using namespace infinicore;
using namespace infinicore::nn;

class MyModel : public Module {
protected:
    INFINICORE_NN_MODULE(Linear, layer1);
    INFINICORE_NN_MODULE(Linear, layer2);
    INFINICORE_NN_MODULE_VEC(Linear, layers);
    INFINICORE_NN_PARAMETER(scaling_factor);
    INFINICORE_NN_BUFFER(cache);

public:
    MyModel() {
        INFINICORE_NN_MODULE_INIT(layer1, 128, 64);
        INFINICORE_NN_MODULE_INIT(layer2, 64, 32);
        INFINICORE_NN_MODULE_VEC_INIT(layers, 3, Linear, 32, 16);
        INFINICORE_NN_PARAMETER_INIT(scaling_factor, ({1}, DataType::F32, Device()));
        INFINICORE_NN_BUFFER_INIT(cache, ({100, 32}, DataType::F32, Device()));
    }
    
    Tensor forward(Tensor &input) {
        Tensor x = layer1_->forward(input);
        x = layer2_->forward(x);
        return x;
    }
    
    // 加载状态字典
    void load_weights(const std::unordered_map<std::string, Tensor> &weights) {
        load_state_dict(weights);
    }
};
```

## 相关链接

- [`nn` 模块概览](../README.md)
- [`Parameter` 参数类](../parameter/README.md)
