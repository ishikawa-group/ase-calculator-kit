"""Device resolution shared by all backends."""

from __future__ import annotations


def resolve_device(device: str = "auto", *, allow_mps: bool = False) -> str | None:
    """Resolve a device string for a calculator.

    Parameters
    ----------
    device:
        ``"auto"`` (default), or an explicit ``"cuda"``, ``"cpu"`` or ``"mps"``.
        Explicit values are passed through unchanged.
    allow_mps:
        When ``True`` (CHGNet on Apple Silicon), ``"auto"`` may resolve to
        ``"mps"``. Most backends (SevenNet, MatterSim, UMA) do not support MPS,
        so they call this with ``allow_mps=False`` and ``"auto"`` falls back to
        ``"cpu"`` when CUDA is unavailable.

    Returns
    -------
    str | None
        The resolved device string. ``"auto"`` resolves to ``"cuda"`` when a
        CUDA device is available, otherwise ``"mps"`` (only if ``allow_mps``),
        otherwise ``"cpu"``.

    Notes
    -----
    ``torch`` is imported lazily so that importing :mod:`ase_umlip_kit` does not
    require a torch installation until a calculator is actually built.
    """
    if device != "auto":
        return device

    try:
        import torch
    except ImportError:
        # No torch yet (e.g. a backend not installed): let the backend surface
        # the appropriate MissingDependencyError. CPU is the safe default.
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"
    if allow_mps and torch.backends.mps.is_built() and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
