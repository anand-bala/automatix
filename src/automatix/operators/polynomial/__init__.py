"""Polynomial-based operator for Alternating Finite Automata.

Provides :class:`PolynomialOperator` and its backend-specific subclasses for
representing AFA transitions and runs as multilinear polynomials over a bounded
distributive lattice (typically Boolean algebra).

Backend variants
----------------
* :class:`NumpyPolynomialOperator` — always available; plain frozen dataclass.
* :class:`JaxPolynomialOperator` — available when ``equinox`` is installed; an
  :class:`equinox.Module` and therefore a JAX PyTree.  ``initial_poly`` is a
  dynamic (learnable) field; structural fields are static.
* :class:`TorchPolynomialOperator` — available when ``torch`` is installed; a
  :class:`torch.nn.Module`.

Usage
-----
Always construct via :py:meth:`PolynomialOperator.from_afa` or
:py:meth:`PolynomialOperator.from_ltl`.
"""

from ._base import PolynomialOperator as PolynomialOperator
from ._base import boolexpr_to_polynomial as boolexpr_to_polynomial
