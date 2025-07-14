
# `InfiniRT` 统一运行时库

## 简介

*InfiniRT* 是一个跨平台统一运行时库，对不同芯片平台的运行时和驱动功能进行了统一封装。

## 编程接口

### 1. 设备管理

#### 初始化设备环境

``` c++
infiniStatus_t infinirtInit();
```

- 初始化所有可用设备类型的运行时环境（如启用的 Ascend）。
- 仅在使用前调用一次。
#### 获得设备所有设备数量

``` c++
infiniStatus_t infinirtGetAllDeviceCount(int *count_array);
```

- `count_array`: 数组长度至少为 [`infiniDevice_t`] 中设备种类数量，每种设备数量会写入数组的相应位置。

#### 获取当前设备信息

``` c++
infiniStatus_t infinirtGetDevice(infiniDevice_t *device_ptr, int *device_id_ptr);
```

- device_ptr: 返回当前设备类型（可为 nullptr）。

- device_id_ptr: 返回当前设备 ID（可为 nullptr）。
#### 获得特定设备数量

``` c++
infiniStatus_t infinirtGetDeviceCount(infiniDevice_t device, int *count);
```

- `device`: 设备种类。
- `count`: 存储设备数量查询结果的指针。

#### 设置当前使用的设备

``` c++
infiniStatus_t infinirtSetDevice(infiniDevice_t device, int device_id);
```

- device: 要设置的设备类型。
- device_id: 要设置的设备 ID。

#### 同步当前设备

``` c++
infiniStatus_t infinirtDeviceSynchronize();
```

- 阻塞直到当前设备完成所有计算。

### 2. 流管理

#### 创建设备流（Stream）

``` c++
infiniStatus_t infinirtStreamCreate(infinirtStream_t *stream_ptr);
```

- stream_ptr: 返回创建的 stream 对象。

#### 销毁设备流

``` c++
infiniStatus_t infinirtStreamDestroy(infinirtStream_t stream);
```

- stream: 需要销毁的 stream 对象。

#### 等待流中所有任务完成

``` c++
infiniStatus_t infinirtStreamSynchronize(infinirtStream_t stream);
```

- stream: 需要同步的 stream。

#### 让一个流等待一个事件

``` c++
infiniStatus_t infinirtStreamWaitEvent(infinirtStream_t stream, infinirtEvent_t event);
```

- stream: 需要等待的流。
- event: 被等待的事件。

### 3. 事件管理
#### 创建事件

``` c++
infiniStatus_t infinirtEventCreate(infinirtEvent_t *event_ptr);
```

- event_ptr: 返回创建的事件。

#### 记录事件

``` c++
infiniStatus_t infinirtEventRecord(infinirtEvent_t event, infinirtStream_t stream);
```

- event: 要记录的事件。
- stream: 事件记录在哪个流上。

#### 查询事件状态

``` c++
infiniStatus_t infinirtEventQuery(infinirtEvent_t event, infinirtEventStatus_t *status_ptr);
```

- event: 需要查询的事件。
- status_ptr: 返回事件状态（已完成 / 未完成等）。

#### 等待事件完成

``` c++
infiniStatus_t infinirtEventSynchronize(infinirtEvent_t event);
```

- event: 要等待的事件。

#### 销毁事件

``` c++
infiniStatus_t infinirtEventDestroy(infinirtEvent_t event);
```

- event: 需要销毁的事件。

### 4. 内存管理

#### 分配设备内存
``` c++
infiniStatus_t infinirtMalloc(void **p_ptr, size_t size);
```

- p_ptr: 返回分配的内存指针。
- size: 要分配的内存大小（单位：字节）。

#### 分配页锁定主机内存

``` c++
infiniStatus_t infinirtMallocHost(void **p_ptr, size_t size);
```

- p_ptr: 返回分配的主机内存指针。

- size: 分配的内存大小（单位：字节）。

#### 释放设备内存

``` c++
infiniStatus_t infinirtFree(void *ptr);
```

- ptr: 需要释放的设备内存指针。

#### 释放页锁定主机内存

``` c++
infiniStatus_t infinirtFreeHost(void *ptr);
```

    ptr: 需要释放的主机内存指针。

#### 同步内存拷贝

``` c++
infiniStatus_t infinirtMemcpy(void *dst, const void *src, size_t size, infinirtMemcpyKind_t kind);
```

- dst: 目标地址。
- src: 源地址。
- size: 拷贝的字节数。
- kind: 枚举类型infinirtMemcpyKind_t 
  其定义如下：
``` c++ 
typedef enum {
    INFINIRT_MEMCPY_H2H = 0, //Host to Host
    INFINIRT_MEMCPY_H2D = 1, //Host to Device
    INFINIRT_MEMCPY_D2H = 2, //Device to Host
    INFINIRT_MEMCPY_D2D = 3, //Device to Device
} infinirtMemcpyKind_t;

```

#### 异步内存拷贝

``` c++
infiniStatus_t infinirtMemcpyAsync(void *dst, const void *src, size_t size, infinirtMemcpyKind_t kind, infinirtStream_t stream);
```

- dst: 目标地址。
- src: 源地址。
- size: 拷贝的字节数。
- kind: 拷贝类型(见[infinirtMemcpy](#同步内存拷贝))。
- stream: 所用的异步流。

#### 异步分配设备内存 (Stream-ordered memory)

``` c++
infiniStatus_t infinirtMallocAsync(void **p_ptr, size_t size, infinirtStream_t stream);
```

- p_ptr: 返回分配的内存指针。

- size: 分配大小（单位：字节）。

- stream: 异步操作所用的流。

#### 异步释放设备内存 (Stream-ordered memory)

``` c++
infiniStatus_t infinirtFreeAsync(void *ptr, infinirtStream_t stream);
```

- ptr: 要释放的设备内存地址。
- stream: 异步释放所用的流。
