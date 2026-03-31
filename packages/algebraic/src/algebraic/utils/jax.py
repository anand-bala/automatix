import equinox as eqx

from algebraic._better_abc import BetterABCMeta


class EqxMeta(type(eqx.Module), BetterABCMeta):
    """Combined metaclass resolving the conflict between equinox's ``_ModuleMeta`` and ``BetterABCMeta``.

    Both are subclasses of ``abc.ABCMeta`` but neither is a subclass of the other,
    so a combined metaclass is required.
    """

    pass
