import importlib.metadata
import importlib.util
from pathlib import Path

_parent_spec = importlib.util.spec_from_file_location("_parent_conf", Path(__file__).resolve().parent.parent / "conf.py")
assert _parent_spec is not None and _parent_spec.loader is not None
_parent_conf = importlib.util.module_from_spec(_parent_spec)
_parent_spec.loader.exec_module(_parent_conf)
globals().update({k: v for k, v in vars(_parent_conf).items() if not k.startswith("_")})

project = "morphata"
package_name = "morphata"
release = importlib.metadata.version(package_name)
