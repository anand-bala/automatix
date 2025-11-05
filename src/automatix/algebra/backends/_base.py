"""Shared utilities for backend implementations.

This module contains helper classes and functions used across multiple backends.
"""

from typing import Union

from typing_extensions import TypeAlias

# Common type aliases
Axis: TypeAlias = Union[None, int, tuple[int, ...]]
Shape: TypeAlias = Union[int, tuple[int, ...]]

__all__ = ["Axis", "Shape"]
