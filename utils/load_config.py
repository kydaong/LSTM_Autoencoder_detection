# Created by aaronkueh on 9/18/2025
from pathlib import Path
import importlib.util

def load_py_config(path: str | Path) -> dict:

    """
    Load a Python config file and return its public attributes as a dict.
    """

    p = Path(path).expanduser().resolve()
    spec = importlib.util.spec_from_file_location(p.stem, str(p))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load Python config from {p}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # executes the config file
    # collect all public names (no leading underscore)
    return {name: getattr(mod, name) for name in dir(mod) if not name.startswith("_")}