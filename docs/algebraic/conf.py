import importlib.metadata
import importlib.util
import typing
from pathlib import Path

if typing.TYPE_CHECKING:
    from sphinx.application import Sphinx

_parent_spec = importlib.util.spec_from_file_location("_parent_conf", Path(__file__).resolve().parent.parent / "conf.py")
assert _parent_spec is not None and _parent_spec.loader is not None
_parent_conf = importlib.util.module_from_spec(_parent_spec)
_parent_spec.loader.exec_module(_parent_conf)
globals().update({k: v for k, v in vars(_parent_conf).items() if not k.startswith("_")})

project = "algebraic"
package_name = "algebraic-arrays"
release = importlib.metadata.version(package_name)

project_metadata = importlib.metadata.metadata(package_name)

description = project_metadata["Summary"]

html_theme_options["description"] = description  # noqa: F821  # ty: ignore[unresolved-reference]

# -- autodoc signature fix for BetterABCMeta ---------------------------------
# BetterABCMeta overrides __call__ with (*args, **kwargs), which Sphinx picks
# up instead of the dataclass __init__. When that happens, read __init__
# directly so the rendered signature shows the actual field names.


def _fix_metaclass_signature(
    app: "Sphinx",
    what: str,
    name: str,
    obj: object,
    options: object,
    signature: str | None,
    return_annotation: str | None,
) -> tuple[str, str | None] | None:
    import inspect

    if what == "class" and signature in ("(*args, **kwargs)", "(*args: Any, **kwargs: Any)"):
        try:
            init = getattr(obj, "__init__", None)
            if init is None:
                return None
            sig = inspect.signature(init)
            params = [p for name_, p in sig.parameters.items() if name_ != "self"]
            new_sig = str(inspect.Signature(params))
            return new_sig, return_annotation
        except (ValueError, TypeError):
            pass
    return None


def setup(app: "Sphinx") -> None:
    app.connect("autodoc-process-signature", _fix_metaclass_signature)
