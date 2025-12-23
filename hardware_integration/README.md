# 硬件接入说明

## 基本功能要求

### 最小实现功能

#### 1. 运行时功能

设备管理：设备初始化，获取设备数量，获取设备信息，设置/激活指定设备，设备同步等功能。  
流管理：流的创建、同步、销毁等功能。  
内存管理：设备及主机内存的分配与释放，同步、异步的内存拷贝等功能。  

#### 2. 底层算子库相关功能

硬件控柄：获取设备信息及基础属性的功能，所需原生算子库的启动和初始化相关功能。  
算子实现：可以直接调用的高性能原生算子库 或 可以用于算子实现的硬件原生编程语言。  

### 进阶功能要求

#### 1. 运行时进阶功能

事件管理：事件的创建、记录、同步、销毁等功能，以及计算事件之间时间差的功能。  

#### 2. 设备间通信的相关功能

通信组初始化、通信组销毁、全组规约(求和、求积、求最大值、求最小值、求均值)  

#### 3. Python算子库可选功能

对Triton、九齿等编程语言的支持。  

## 接口实现要求

### 最小实现

#### 1. [`InfiniRT`]：实现统一运行时库中的设备管理、流管理、内存管理相关功能

##### 1-1. 设备管理

基础功能/接口要求：
- 设备初始化`infinirtInit()`
- 获取设备数量`infinirtGetAllDeviceCount(int *count_array)`
- 获取设备信息`infinirtGetDevice(infiniDevice_t *device_ptr, int *device_id_ptr)`
- 设置指定设备`infinirtSetDevice(infiniDevice_t device, int device_id)`
- 同步当前设备`infinirtDeviceSynchronize()`

##### 1-2. 流管理

基础功能/接口要求：
- 流创建`infinirtStreamCreate(infinirtStream_t *stream_ptr)`
- 流销毁`infinirtStreamDestroy(infinirtStream_t stream)`
- 流同步`infinirtStreamSynchronize(infinirtStream_t stream)`

##### 1-3. 内存管理

基础功能/接口要求：
- 设备内存分配`infinirtMalloc(void **p_ptr, size_t size)`
- 主机内存分配`infinirtMallocHost(void **p_ptr, size_t size)`
- 设备内存释放`infinirtFree(void *ptr)`
- 主机内存释放`infinirtFreeHost(void *ptr)`
- 同步内存拷贝`infinirtMemcpy(void *dst, const void *src, size_t size, infinirtMemcpyKind_t kind)`
- 异步内存拷贝`infinirtMemcpyAsync(void *dst, const void *src, size_t size, infinirtMemcpyKind_t kind, infinirtStream_t stream)`

#### 2. [`InfiniOP`]：实现统一C++算子库中硬件控柄相关功能，以及具体的算子实现

##### 2-1. 硬件控柄 - [`infiniop/handle`]

- 创建控柄`infiniopCreateHandle(infiniopHandle_t *handle_ptr)`
- 销毁控柄`infiniopDestroyHandle(infiniopHandle_t handle)`

##### 2-2. 算子实现 - [`infiniop/ops`]

###### 实现C++算子库中的算子通常有2种选择

a. 使用原生C++算子库直接接入infiniop算子实现  
b. 使用原生硬件编程语言接入高度定制化的算子实现  

###### 单个算子<OP>需要实现的接口：
- 描述符创建
```c++
infiniStatus_t infiniopCreate<OP>Descriptor(
    infiniopHandle_t handle,
    infiniop<OP>Descriptor_t *desc,
    infiniopTensorDescriptor_t desc,
    ...
);
``` 
- 工作空间获取
```c++
infiniStatus_t infiniopGet<OP>Workspace(
    infiniop<OP>Descriptor_t desc,
    size_t *size
);
``` 
- 计算操作
```c++
infiniStatus_t infiniop<OP>(
    infiniop<OP>Descriptor_t desc,
    void *workspace,
    size_t workspace_size,
    infiniopTensorDescriptor_t desc,
    ...
);
```
- 描述符销毁
```c++
infiniStatus_t infiniopDestroy<OP>Descriptor(
    infiniop<OP>Descriptor_t desc
);
```

### 进阶实现

#### 1. [`InfiniRT`]：实现统一运行时库中的事件管理相关功能

- 用于设备端性能统计。

接口要求：
- 事件创建`infinirtEventCreate(infinirtEvent_t *event_ptr)`
- 事件记录`infinirtEventRecord(infinirtEvent_t event, infinirtStream_t stream)`
- 事件同步`infinirtEventSynchronize(infinirtEvent_t event)`
- 事件销毁`infinirtEventDestroy(infinirtEvent_t event)`
- 事件时间差计算`infinirtEventElapsedTime(float *ms_ptr, infinirtEvent_t start, infinirtEvent_t end)`

#### 2. [`InfiniCCL`]：实现设备间通信的相关功能

- 用于分布式计算。

接口要求：
- 通信组初始化
``` c++
infiniStatus_t infinicclCommInitAll(
    infiniDevice_t device_type,
    infinicclComm_t *comms,
    int ndevice,
    const int *device_ids);
```
- 通信组销毁
``` c++
infiniStatus_t infinicclCommDestroy(infinicclComm_t comm);
```
- 全组规约
``` c++
infiniStatus_t infinicclAllReduce(
    void *sendbuf,
    void *recvbuf,
    size_t count,
    infiniDtype_t datatype,
    infinicclReduceOp_t op,
    infinicclComm_t comm,
    infinirtStream_t stream);
```

#### 3. [`infinicore/ops`]：实现Python算子库。

用于在Python端通过简洁的调用方式快速完成计算。通常有3种实现方式：

a. 通过Python及pybind11接入底层C++算子库中的算子实现。

b. 使用Triton或九齿等语言实现通用的Python算子。

[`InfiniRT`]:/infinirt/README.md
[`InfiniOP`]:/infiniop/README.md
[`InfiniCCL`]:/infiniccl/README.md
[`infiniop/handle`]:/infiniop/handle/README.md
[`infiniop/ops`]:/infiniop/ops/README.md
[`infinicore/ops`]:/python/ops/README.md
