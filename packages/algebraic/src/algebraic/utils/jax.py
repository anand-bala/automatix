import equinox as eqx

from algebraic._better_abc import BetterABCMeta


class EqxMeta(type(eqx.Module), BetterABCMeta):
    """
    Combined metaclass to resolve conflict between equinox's _ModuleMeta and BetterABCMeta.
    Both are subclasses of abc.ABCMeta but neither is a subclass of the other,
    so we need a combined metaclass.
    """

    pass
