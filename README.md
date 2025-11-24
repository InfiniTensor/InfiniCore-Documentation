# InfiniCore v0.2.0

## 项目简介

*InfiniCore* 是一个跨平台统一的人工智能编程框架，暴露 C++ 和 Python 两种编程接口，提供易于调用的、跨平台统一的高性能张量编程功能。同时它还对外暴露 *InfiniCore-C* 编程接口，为不同芯片平台的底层功能（包括计算、运行时、通信等）提供统一 C 语言接口封装。

项目地址：<https://github.com/InfiniTensor/InfiniCore>

## 文档目录

### InfiniCore
- [`Python APIs`](python/README.md)

- [`C++ APIs`]

### InfiniCore-C

- [`InfiniRT`]：统一运行时库，提供基础的运行时功能，包括线程、内存管理、同步、事件通知等。

- [`InfiniOP`]：统一算子库，提供各类基于张量的高性能算子计算功能。

- [`InfiniCCL`]：统一集合通信库，提供常用的集合通信功能，包括点对点、广播、聚合等。


[`Python APIs`]:python/README.md
[`C++ APIs`]:README.md
[`InfiniRT`]:/infinirt/README.md
[`InfiniOP`]:/infiniop/README.md
[`InfiniCCL`]:/infiniccl/README.md
