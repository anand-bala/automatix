"""Matrix-based weighted automaton operator.

Provides :class:`MatrixOperator` and its backend-specific subclasses for
constructing weighted finite-word automaton operators from an NFA and a weight
function.

Backend variants
----------------
* :class:`NumpyMatrixOperator` — always available; plain frozen dataclass.
* :class:`JaxMatrixOperator` — available when ``equinox`` is installed; an
  :class:`equinox.Module` and therefore a JAX PyTree.  The ``initial_weights``,
  ``final_weights``, and ``weight_function`` fields are dynamic (learnable)
  leaves; ``semiring`` and ``_transition_graph`` are static.
* :class:`TorchMatrixOperator` — available when ``torch`` is installed; a
  :class:`torch.nn.Module`.  If ``weight_function`` is itself an
  :class:`~torch.nn.Module`, it is registered as a submodule and its parameters
  appear in :py:meth:`~torch.nn.Module.parameters`.

Usage
-----
Always construct via :py:meth:`MatrixOperator.make`; never instantiate the
backend-specific classes directly.
"""

from ._base import MatrixOperator as MatrixOperator
