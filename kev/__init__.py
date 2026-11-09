"""Kev: `from kev import Kev`. The import is lazy so that modules that only need kev.api or kev.model (the Hugging Face Space
vendors those two) do not pull in the loader."""


def __getattr__(name):
    if name == "Kev":
        from .runtime import Kev
        return Kev
    raise AttributeError(f"module 'kev' has no attribute {name!r}")
