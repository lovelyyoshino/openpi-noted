import contextlib
import functools as ft
import inspect
from typing import TypeAlias, TypeVar, cast

import beartype
import jax
import jax._src.tree_util as private_tree_util
import jax.core
from jaxtyping import Array  # noqa: F401
from jaxtyping import ArrayLike
from jaxtyping import Bool  # noqa: F401
from jaxtyping import DTypeLike  # noqa: F401
from jaxtyping import Float
from jaxtyping import Int  # noqa: F401
from jaxtyping import Key  # noqa: F401
from jaxtyping import Num  # noqa: F401
from jaxtyping import PyTree
from jaxtyping import Real  # noqa: F401
from jaxtyping import UInt8  # noqa: F401
from jaxtyping import config
from jaxtyping import jaxtyped
import jaxtyping._decorator

# 对 jaxtyping 进行补丁，以处理 https://github.com/patrick-kidger/jaxtyping/issues/277 的问题。
# 问题在于自定义 PyTree 节点有时会被初始化为任意类型（例如 `jax.ShapeDtypeStruct`，
# `jax.Sharding`，甚至是 <object>），这是由于 JAX 跟踪操作导致的。该补丁在堆栈跟踪包含 
# `jax._src.tree_util` 时跳过类型检查，这种情况只发生在树解除平坦化期间。
_original_check_dataclass_annotations = jaxtyping._decorator._check_dataclass_annotations  # noqa: SLF001


def _check_dataclass_annotations(self, typechecker):
    """ 检查数据类注释的函数，如果当前栈帧的全局名称不是 jax._src.tree_util 或 flax.nnx.transforms.compilation，则执行原始注释检查。

    Args:
        self: 当前实例，用于存取相关信息。
        typechecker: 类型检查器，用于验证类型。

    Returns:
        如果不在指定的框架内则返回原始的数据类注释检查，否则返回 None。
    """
    if not any(
        frame.frame.f_globals["__name__"] in {"jax._src.tree_util", "flax.nnx.transforms.compilation"}
        for frame in inspect.stack()
    ):
        return _original_check_dataclass_annotations(self, typechecker)
    return None


jaxtyping._decorator._check_dataclass_annotations = _check_dataclass_annotations  # noqa: SLF001

KeyArrayLike: TypeAlias = jax.typing.ArrayLike  # 定义 KeyArrayLike 为 jax.typing 中的 ArrayLike
Params: TypeAlias = PyTree[Float[ArrayLike, "..."]]  # 定义 Params 为泛型的 PyTree，其中值为 Float 类型的 ArrayLike

T = TypeVar("T")  # 声明一个类型变量 T


# 运行时类型检查装饰器
def typecheck(t: T) -> T:
    """ 用于对输入类型进行检查的装饰器。

    Args:
        t: 要进行检查的类型标记。

    Returns:
        返回经过类型检查的对象 cast 到类型 T。
    """
    return cast(T, ft.partial(jaxtyped, typechecker=beartype.beartype)(t))


@contextlib.contextmanager
def disable_typechecking():
    """ 上下文管理器，用于禁用类型检查。

    Yields:
        在 yield 语句前设置禁止类型检查为 True，然后在上下文结束后恢复初始状态。
    """
    initial = config.jaxtyping_disable
    config.update("jaxtyping_disable", True)  # noqa: FBT003
    yield
    config.update("jaxtyping_disable", initial)


def check_pytree_equality(*, expected: PyTree, got: PyTree, check_shapes: bool = False, check_dtypes: bool = False):
    """ 检查两个 PyTree 是否具有相同结构，并可选地检查形状和数据类型。创建比 naively 使用 `jax.tree.map`
    更加友好的错误信息，尤其是在不同结构的 PyTree 比较时。

    Args:
        expected: 期望的 PyTree。
        got: 实际获得的 PyTree。
        check_shapes: 是否检查形状相等的标志。
        check_dtypes: 是否检查数据类型相等的标志。

    Raises:
        ValueError: 当两个 PyTree 的结构不一致或根据所需的检查失败时抛出此异常。
    """

    if errors := list(private_tree_util.equality_errors(expected, got)):
        raise ValueError(
            "PyTrees have different structure:\n"
            + (
                "\n".join(
                    f"   - at keypath '{jax.tree_util.keystr(path)}': expected {thing1}, got {thing2}, so {explanation}.\n"
                    for path, thing1, thing2, explanation in errors
                )
            )
        )

    if check_shapes or check_dtypes:

        def check(kp, x, y):
            """ 辅助函数检查每个键路径对应的项是否匹配 shapes 和 dtypes。

            Args:
                kp: 当前的键路径。
                x: 来自预期的 PyTree 的值。
                y: 来自实际的 PyTree 的值。

            Raises:
                ValueError: 当它们的形状或数据类型不匹配时抛出异常。
            """
            if check_shapes and x.shape != y.shape:
                raise ValueError(f"Shape mismatch at {jax.tree_util.keystr(kp)}: expected {x.shape}, got {y.shape}")

            if check_dtypes and x.dtype != y.dtype:
                raise ValueError(f"Dtype mismatch at {jax.tree_util.keystr(kp)}: expected {x.dtype}, got {y.dtype}")

        jax.tree_util.tree_map_with_path(check, expected, got)  # 遍历图并应用检查
