
# `infiniccl`统一集合通信库

## 简介

  `InfiniCCL`：统一集合通信库，提供常用的集合通信功能，包括点对点、广播、聚合等。

## 编程接口
#### 初始化通信组

``` c++
infiniStatus_t infinicclCommInitAll(
    infiniDevice_t device_type,
    infinicclComm_t *comms,
    int ndevice,
    const int *device_ids);
```

- device_type: 设备类型，枚举类型 infiniDevice_t，例如 INFINI_DEVICE_NVIDIA。
- comms: 输出通信组句柄数组，长度为 ndevice。
- ndevice: 设备数量。
- device_ids: 设备 ID 数组，长度为 ndevice，每个 ID 对应 device_type 类型下的一个设备。
- 说明：为指定设备类型的若干设备创建通信上下文，用于后续 AllReduce 等 collective 操作。每个设备对应一个通信句柄。

#### 销毁通信组

``` c++
infiniStatus_t infinicclCommDestroy(infinicclComm_t comm);
```

- comm: 需要销毁的通信组句柄。
- 说明：释放通过 infinicclCommInitAll 创建的通信资源。

#### AllReduce 操作

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

- sendbuf: 输入缓冲区，存放需要参与 AllReduce 的本地数据。
- recvbuf: 输出缓冲区，存放归约后的结果数据。
- count: 元素个数（而不是字节数）。
- datatype: [枚举类型infiniDtype_t](/common/dtype/README.md)。
- op: 归约操作类型，枚举类型 infinicclReduceOp_t。 
  其定义如下: 
```
  typedef enum {
    INFINICCL_SUM = 0,
    INFINICCL_PROD = 1,
    INFINICCL_MAX = 2,
    INFINICCL_MIN = 3,
    INFINICCL_AVG = 4,
} infinicclReduceOp_t;
```
- comm: 通信组句柄。
- stream: 指定操作执行在哪个 infinirtStream_t 流上。
- 说明：在通信组内所有设备上执行 AllReduce 操作。各设备的 recvbuf 将接收到相同的归约结果。
