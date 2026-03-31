# This module is adapted from equinox (https://github.com/patrick-kidger/equinox),
# licensed under the MIT License. See the equinox repository for the original source.

"""Introduces `AbstractVar` and `AbstractClassVar` type annotations, which can be used
to mark abstract instance attributes and abstract class attributes.

Provides `BetterABCMeta` (an enhanced `abc.ABCMeta`) and `better_dataclass` (an enhanced
`dataclasses.dataclass`) that respect these type annotations.
"""

from __future__ import annotations

import abc
import dataclasses
import inspect
from collections.abc import Callable
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    ClassVar,
    Generic,
    NoReturn,
    TypeAlias,
    TypeVar,
    get_args,
    get_origin,
)

from typing_extensions import dataclass_transform, overload

_T = TypeVar("_T")
_C = TypeVar("_C")


if TYPE_CHECKING:
    AbstractVar: TypeAlias = Annotated[_T, "AbstractVar"]
    from typing import ClassVar as AbstractClassVar
else:

    class AbstractVar(Generic[_T]):
        """Used to mark an abstract instance attribute, along with its type. Used as:
        ```python
        from algebraic._better_abc import BetterABCMeta, better_dataclass, AbstractVar

        @better_dataclass()
        class Foo(metaclass=BetterABCMeta):
            attr: AbstractVar[bool]
        ```

        An `AbstractVar[T]` must be overridden by an attribute annotated with
        `AbstractVar[T]`, `AbstractClassVar[T]`, `ClassVar[T]`, `T`, or a property
        returning `T`.

        This makes `AbstractVar` useful when you just want to assert that you can access
        `self.attr` on a subclass, regardless of whether it's an instance attribute,
        class attribute, property, etc.

        Attempting to instantiate a class with an unoverridden `AbstractVar` will raise
        an error.

        Example::

            from algebraic._better_abc import BetterABCMeta, better_dataclass, AbstractVar

            @better_dataclass()
            class AbstractX(metaclass=BetterABCMeta):
                attr1: int
                attr2: AbstractVar[bool]

            @better_dataclass()
            class ConcreteX(AbstractX):
                attr2: bool

            ConcreteX(attr1=1, attr2=True)

        Note:
            `AbstractVar` does not create a dataclass field. This affects the order of
            `__init__` arguments. E.g.::

                @better_dataclass()
                class AbstractX(metaclass=BetterABCMeta):
                    attr1: AbstractVar[bool]

                @better_dataclass()
                class ConcreteX(AbstractX):
                    attr2: str
                    attr1: bool

            should be called as `ConcreteX(attr2, attr1)`.
        """

    class AbstractClassVar(Generic[_T]):
        """Used to mark an abstract class attribute, along with its type. Used as:
        ```python
        from algebraic._better_abc import BetterABCMeta, better_dataclass, AbstractClassVar

        @better_dataclass()
        class Foo(metaclass=BetterABCMeta):
            attr: AbstractClassVar[bool]
        ```

        An `AbstractClassVar[T]` can be overridden by an attribute annotated with
        `AbstractClassVar[T]`, or `ClassVar[T]`. This makes `AbstractClassVar` useful
        when you want to assert that you can access `cls.attr` on a subclass.

        Attempting to instantiate a class with an unoverridden `AbstractClassVar` will
        raise an error.

        Example::

            from algebraic._better_abc import BetterABCMeta, better_dataclass, AbstractClassVar
            from typing import ClassVar

            @better_dataclass()
            class AbstractX(metaclass=BetterABCMeta):
                attr1: int
                attr2: AbstractClassVar[bool]

            @better_dataclass()
            class ConcreteX(AbstractX):
                attr2: ClassVar[bool] = True

            ConcreteX(attr1=1)

        Note:
            `AbstractClassVar` does not create a dataclass field. This affects the order
            of `__init__` arguments. E.g.::

                @better_dataclass()
                class AbstractX(metaclass=BetterABCMeta):
                    attr1: AbstractClassVar[bool]

                @better_dataclass()
                class ConcreteX(AbstractX):
                    attr2: str
                    attr1: ClassVar[bool] = True

            should be called as `ConcreteX(attr2)`.
        """


def _process_annotation(annotation: object) -> tuple[bool, bool]:
    if isinstance(annotation, str):
        if annotation.startswith("AbstractVar[") or annotation.startswith("AbstractClassVar["):
            raise NotImplementedError("Stringified abstract annotations are not supported")
        else:
            is_abstract = False
            is_class = annotation.startswith("ClassVar[")
            return is_abstract, is_class
    else:
        if annotation in (AbstractVar, AbstractClassVar):
            raise TypeError("Cannot use unsubscripted `AbstractVar` or `AbstractClassVar`.")
        elif get_origin(annotation) is AbstractVar:
            if len(get_args(annotation)) != 1:
                raise TypeError("`AbstractVar` can only have a single argument.")
            is_abstract = True
            is_class = False
        elif get_origin(annotation) is AbstractClassVar:
            if len(get_args(annotation)) != 1:
                raise TypeError("`AbstractClassVar` can only have a single argument.")
            is_abstract = True
            is_class = True
        elif get_origin(annotation) is ClassVar:
            is_abstract = False
            is_class = True
        else:
            is_abstract = False
            is_class = False
        return is_abstract, is_class


class BetterABCMeta(abc.ABCMeta):
    __abstractvars__: frozenset[str]
    __abstractclassvars__: frozenset[str]

    def register(cls, subclass: type) -> NoReturn:
        del subclass
        raise ValueError

    def __new__(mcs, name: str, bases: tuple[type, ...], namespace: dict[str, Any], /, **kwargs: Any) -> BetterABCMeta:  # noqa: ANN401
        cls = super().__new__(mcs, name, bases, namespace, **kwargs)

        # We don't try and check that our AbstractVars and AbstractClassVars are
        # consistently annotated across `cls` and each element of `bases`. Python just
        # doesn't really provide any way of checking that two hints are compatible.
        # (Subscripted generics make this complicated!)

        abstract_vars = set()
        abstract_class_vars = set()
        for kls in reversed(cls.__mro__):
            ann = inspect.get_annotations(kls)
            for attr_name, annotation in ann.items():
                is_abstract, is_class = _process_annotation(annotation)
                if is_abstract:
                    if is_class:
                        if attr_name in kls.__dict__:
                            raise TypeError(f"Abstract class attribute {attr_name} cannot have value")
                        abstract_vars.discard(attr_name)
                        abstract_class_vars.add(attr_name)
                    else:
                        if attr_name in kls.__dict__:
                            raise TypeError(f"Abstract attribute {attr_name} cannot have value")
                        # If it's already an abstract class var, then superfluous to
                        # also consider it an abstract var.
                        if attr_name not in abstract_class_vars:
                            abstract_vars.add(attr_name)
                else:
                    abstract_class_vars.discard(attr_name)
                    if not is_class:
                        abstract_vars.discard(attr_name)
            for attr_name in kls.__dict__.keys():
                abstract_vars.discard(attr_name)
                abstract_class_vars.discard(attr_name)
        cls.__abstractvars__ = frozenset(abstract_vars)  # pyright: ignore
        cls.__abstractclassvars__ = frozenset(abstract_class_vars)  # pyright: ignore
        return cls

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        __tracebackhide__ = True
        if len(cls.__abstractclassvars__) > 0:  # pyright: ignore
            abstract_class_vars = set(cls.__abstractclassvars__)  # pyright: ignore
            raise TypeError(
                f"Can't instantiate abstract class {cls.__name__} with abstract class attributes {abstract_class_vars}"
            )
        self = super().__call__(*args, **kwargs)
        if len(cls.__abstractvars__) > 0:  # pyright: ignore
            abstract_class_vars = set(cls.__abstractvars__)  # pyright: ignore
            raise TypeError(f"Can't instantiate abstract class {cls.__name__} with abstract attributes {abstract_class_vars}")
        return self


@overload
def better_dataclass(cls: type[_C], /, **kwargs: Any) -> type[_C]: ...  # noqa: ANN401
@overload
def better_dataclass(cls: None = None, /, **kwargs: Any) -> Callable[[type[_C]], type[_C]]: ...  # noqa: ANN401
@dataclass_transform()
def better_dataclass(cls: type[_C] | None = None, /, **kwargs: Any) -> type[_C] | Callable[[type[_C]], type[_C]]:  # noqa: ANN401
    def make_dataclass(kls: type[_C]) -> type[_C]:
        try:
            annotations = inspect.get_annotations(kls)
        except KeyError:
            kls = dataclasses.dataclass(**kwargs)(kls)
        else:
            new_annotations = dict(annotations)
            for attr_name, annotation in annotations.items():
                is_abstract, _ = _process_annotation(annotation)
                if is_abstract:
                    new_annotations.pop(attr_name)
            kls.__annotations__ = new_annotations
            kls = dataclasses.dataclass(**kwargs)(kls)
            kls.__annotations__ = annotations
        return kls

    if cls is None:
        return make_dataclass
    return make_dataclass(cls)


@overload
def frozen(cls: type[_C], /, **kwargs: Any) -> type[_C]: ...  # noqa: ANN401
@overload
def frozen(cls: None = None, /, **kwargs: Any) -> Callable[[type[_C]], type[_C]]: ...  # noqa: ANN401
@dataclass_transform()
def frozen(cls: type[_C] | None = None, /, **kwargs: Any) -> type[_C] | Callable[[type[_C]], type[_C]]:  # noqa: ANN401
    """Shorthand for ``better_dataclass(frozen=True, ...)``."""
    kwargs.setdefault("frozen", True)
    return better_dataclass(cls, **kwargs)
