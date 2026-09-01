"""Runtime compatibility shims for running pinned SAM 3.1 on this host.

The official SAM 3.1 README requires torch >= 2.7.  This host is Ubuntu 18.04
(glibc 2.27) and the available CUDA wheel for that platform is torch 2.5.1.
One observable incompatibility is that CUDA ``torch.sort`` / ``argsort`` does
not support bool dtype until a later PyTorch release.

The official code only uses ``argsort`` to order a boolean keep-mask
(``sam3_multiplex_base.py:542``).  Sorting a bool mask is exactly equivalent
to sorting the same mask cast to uint8.  We therefore install a minimal,
semantically identical monkeypatch inside our own adapter layer.  No
third-party source file is modified; the shim is documented in the N0 audit.
"""

import torch

_orig_tensor_argsort = torch.Tensor.argsort
_orig_tensor_sort = torch.Tensor.sort
_orig_torch_argsort = torch.argsort
_orig_torch_sort = torch.sort


def _cast_bool_argsort(tensor, dim=-1, descending=False, *, stable=False):
    if tensor.dtype == torch.bool:
        return tensor.to(torch.uint8).argsort(
            dim=dim, descending=descending, stable=stable
        )
    return _orig_tensor_argsort(tensor, dim=dim, descending=descending, stable=stable)


def _cast_bool_sort(tensor, dim=-1, descending=False, *, stable=False):
    if tensor.dtype == torch.bool:
        values, indices = tensor.to(torch.uint8).sort(
            dim=dim, descending=descending, stable=stable
        )
        return values.to(torch.bool), indices
    return _orig_tensor_sort(tensor, dim=dim, descending=descending, stable=stable)


def _torch_argsort(input, dim=-1, descending=False, *, stable=False):
    if input.dtype == torch.bool:
        return input.to(torch.uint8).argsort(
            dim=dim, descending=descending, stable=stable
        )
    return _orig_torch_argsort(input, dim=dim, descending=descending, stable=stable)


def _torch_sort(input, dim=-1, descending=False, *, stable=False):
    if input.dtype == torch.bool:
        values, indices = input.to(torch.uint8).sort(
            dim=dim, descending=descending, stable=stable
        )
        return values.to(torch.bool), indices
    return _orig_torch_sort(input, dim=dim, descending=descending, stable=stable)


def install_compat_shims() -> None:
    """Install bool-sort compatibility shims (idempotent)."""
    torch.Tensor.argsort = _cast_bool_argsort
    torch.Tensor.sort = _cast_bool_sort
    torch.argsort = _torch_argsort
    torch.sort = _torch_sort


install_compat_shims()
