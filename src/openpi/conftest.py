import os

import pynvml
import pytest


def set_jax_cpu_backend_if_no_gpu() -> None:
    """
    如果系统中没有找到可用的 GPU，则将 JAX 的计算后端设置为 CPU。

    此函数尝试初始化 NVML（NVIDIA Management Library），如果初始化失败，
    则认为系统中没有可用的 GPU，并将环境变量 JAX_PLATFORMS 设置为 'cpu'。
    """
    try:
        # 初始化 NVML 以检测系统中是否有可用的 GPU
        pynvml.nvmlInit()
        # 关闭 NVML，释放资源
        pynvml.nvmlShutdown()
    except pynvml.NVMLError:
        # 捕获 NVML 初始化错误，表明没有找到可用的 GPU
        # 设置 JAX 计算后端为 CPU
        os.environ["JAX_PLATFORMS"] = "cpu"


def pytest_configure(config: pytest.Config) -> None:
    """
    在 pytest 配置阶段调用的钩子函数。

    此函数会在 pytest 开始配置时被调用，用于执行一些初始化操作。
    具体来说，它会调用 set_jax_cpu_backend_if_no_gpu 函数，
    以确保如果系统中没有可用的 GPU，JAX 的计算后端会被设置为 CPU。

    Args:
        config (pytest.Config): pytest 的配置对象，当前未被使用。
    """
    # 调用 set_jax_cpu_backend_if_no_gpu 函数
    # 该函数会检查系统中是否有可用的 GPU，如果没有则将 JAX 的计算后端设置为 CPU
    set_jax_cpu_backend_if_no_gpu()
