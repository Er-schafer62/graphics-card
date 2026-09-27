"""Kernels that ship with the replica, written in GCN-style assembly (``*.s``)."""

from __future__ import annotations

from importlib import resources


def builtin_kernel_names() -> list[str]:
    return sorted(p.name[:-2] for p in resources.files(__name__).iterdir() if p.name.endswith(".s"))


def builtin_kernel(name: str) -> str:
    """Return the assembly source of a built-in kernel."""
    path = resources.files(__name__).joinpath(f"{name}.s")
    if not path.is_file():
        raise KeyError(f"no built-in kernel '{name}' (have: {', '.join(builtin_kernel_names())})")
    return path.read_text()
