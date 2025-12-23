# `infinicore::context` 运行时上下文

运行时上下文模块，实现位于 `InfiniCore/src/infinicore/context/`，提供设备管理、内存分配、流同步等核心运行时功能。

## 模块结构

`context` 模块包含以下主要组件：

- **ContextImpl** (`context_impl.hpp/cc`)：上下文实现类，管理所有设备的运行时实例
- **Runtime** (`runtime/runtime.hpp/cc`)：单个设备的运行时封装，管理设备流、算子句柄和内存分配器
- **MemoryAllocator** (`allocators/memory_allocator.hpp`)：内存分配器基类
- **HostAllocator** (`allocators/host_allocator.hpp/cc`)：主机内存分配器
- **DeviceCachingAllocator** (`allocators/device_caching_allocator.hpp/cc`)：设备内存缓存分配器
- **DevicePinnedHostAllocator** (`allocators/device_pinned_allocator.hpp/cc`)：设备固定主机内存分配器

## ContextImpl 类

`ContextImpl` 是单例类，管理所有设备的运行时实例。定义于 `InfiniCore/src/infinicore/context/context_impl.hpp`。

### 主要功能

- 维护一个运行时表（`runtime_table_`），为每种设备类型和索引存储 `Runtime` 实例
- 使用线程局部存储（`thread_local`）跟踪当前线程的活跃运行时
- 提供设备切换和查询接口

### 主要方法

- `singleton()`：获取单例实例
- `getCurrentRuntime()`：获取当前线程的活跃运行时（懒初始化）
- `getCpuRuntime()`：获取 CPU 运行时
- `setDevice(device)`：设置当前设备，懒初始化运行时
- `getDeviceCount(type)`：获取指定类型的设备数量

### 初始化行为

- 构造函数中查询所有可用设备数量
- 优先使用第一个非 CPU 设备作为默认运行时
- 如果没有非 CPU 设备，则回退到 CPU

## Runtime 类

`Runtime` 封装单个设备的运行时状态，定义于 `InfiniCore/src/infinicore/context/runtime/runtime.hpp`。

### 主要成员

- `device_`：关联的设备
- `stream_`：设备的默认流（`infinirtStream_t`）
- `infiniop_handle_`：InfiniOP 算子库句柄
- `device_memory_allocator_`：设备内存分配器
- `pinned_host_memory_allocator_`：固定主机内存分配器（仅非 CPU 设备）

### 主要方法

- `activate()`：激活此运行时（设置当前设备）
- `device()`：获取关联的设备
- `stream()`：获取默认流
- `infiniopHandle()`：获取 InfiniOP 句柄
- `syncStream()` / `syncDevice()`：同步流/设备
- `allocateMemory(size)`：分配设备内存
- `allocatePinnedHostMemory(size)`：分配固定主机内存
- 内存拷贝方法：`memcpyH2D`、`memcpyD2H`、`memcpyD2D`
- 事件管理方法：`createEvent`、`recordEvent`、`synchronizeEvent` 等

### 分配器选择

- CPU 设备：使用 `HostAllocator`
- 非 CPU 设备：使用 `DeviceCachingAllocator` 和 `DevicePinnedHostAllocator`

## 内存分配器

### MemoryAllocator 基类

定义于 `InfiniCore/src/infinicore/context/allocators/memory_allocator.hpp`，提供统一的内存分配接口：

```cpp
class MemoryAllocator {
public:
    virtual std::byte *allocate(size_t size) = 0;
    virtual void deallocate(std::byte *ptr) = 0;
};
```

### HostAllocator

主机内存分配器，定义于 `InfiniCore/src/infinicore/context/allocators/host_allocator.hpp`。

- 使用标准 `new` / `delete` 分配主机内存
- 用于 CPU 设备的内存分配

### DeviceCachingAllocator

设备内存缓存分配器，定义于 `InfiniCore/src/infinicore/context/allocators/device_caching_allocator.hpp`。

- 用于非 CPU 设备的设备内存分配
- 通过 InfiniRT 接口分配设备内存

### DevicePinnedHostAllocator

设备固定主机内存分配器，定义于 `InfiniCore/src/infinicore/context/allocators/device_pinned_allocator.hpp`。

- 用于分配固定（pinned）主机内存，便于设备与主机之间的快速数据传输
- 包含垃圾回收队列（`gc_queue_`），用于延迟释放内存

## context 命名空间 API

`context` 命名空间（定义于 `InfiniCore/include/infinicore/context/context.hpp`）提供用户友好的 API，内部调用 `ContextImpl` 和 `Runtime`：

### 设备管理

```cpp
namespace infinicore::context {
    void setDevice(Device device);
    Device getDevice();
    size_t getDeviceCount(Device::Type type);
}
```

### 流管理

```cpp
infinirtStream_t getStream();
```

### 内存分配

```cpp
std::shared_ptr<Memory> allocateMemory(size_t size);
std::shared_ptr<Memory> allocateHostMemory(size_t size);
std::shared_ptr<Memory> allocatePinnedHostMemory(size_t size);
```

### 内存拷贝

```cpp
void memcpyH2D(void *dst, const void *src, size_t size, bool async = true);
void memcpyD2H(void *dst, const void *src, size_t size);
void memcpyD2D(void *dst, const void *src, size_t size, bool async = true);
void memcpyH2H(void *dst, const void *src, size_t size);
```

### 同步操作

```cpp
void syncStream();
void syncDevice();
```

### 事件管理

```cpp
infinirtEvent_t createEvent();
infinirtEvent_t createEventWithFlags(uint32_t flags);
void recordEvent(infinirtEvent_t event, infinirtStream_t stream = nullptr);
bool queryEvent(infinirtEvent_t event);
void synchronizeEvent(infinirtEvent_t event);
void destroyEvent(infinirtEvent_t event);
float elapsedTime(infinirtEvent_t start, infinirtEvent_t end);
void streamWaitEvent(infinirtStream_t stream, infinirtEvent_t event);
```

### 算子句柄

```cpp
infiniopHandle_t getInfiniopHandle(Device device);
```

## 使用示例

```cpp
#include <infinicore.hpp>

using namespace infinicore;

// 设置设备（内部会创建或激活对应的 Runtime）
Device device(Device::Type::NVIDIA, 0);
context::setDevice(device);

// 分配内存（通过 Runtime 的分配器）
std::shared_ptr<Memory> mem = context::allocateMemory(1024 * 1024);

// 创建张量（内部使用 context 分配内存）
Tensor tensor = Tensor::empty({1000, 1000}, DataType::F32, device);

// 同步设备
context::syncDevice();
```

## 实现细节

- `ContextImpl` 使用单例模式，进程内全局唯一
- `Runtime` 使用线程局部存储，每个线程维护独立的当前运行时指针
- 内存分配器通过 `Memory` 的删除器（deleter）自动管理内存生命周期
- 设备切换时会自动激活对应的 `Runtime`，确保操作在正确的设备上执行

## 相关链接

- [`Device` 文档](../device/README.md)
- [`Memory` 文档](../memory/README.md)
- [`DeviceEvent` 文档](../device_event/README.md)
- [`Tensor` 文档](../tensor/README.md)
