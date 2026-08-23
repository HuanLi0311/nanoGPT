#!/usr/bin/env python3
import runpy
import sys
import builtins

import torch


def patch_torch_23() -> None:
    import torch.distributed as dist
    import torch.distributed.tensor as dist_tensor

    if not hasattr(dist, "DeviceMesh"):
        from torch.distributed.device_mesh import DeviceMesh

        dist.DeviceMesh = DeviceMesh
    if not hasattr(dist_tensor, "DTensor"):
        from torch.distributed._tensor import DTensor

        dist_tensor.DTensor = DTensor
        from torch.distributed._tensor.placement_types import DTensorSpec

        builtins.DTensorSpec = DTensorSpec
    if hasattr(torch.nested, "nested_tensor_from_jagged"):
        return

    from torch.nested._internal.nested_tensor import (
        NestedTensor,
        nested_view_from_values_offsets,
        nested_view_from_values_offsets_lengths,
    )

    def nested_tensor_from_jagged(values, offsets, lengths=None, ragged_idx=1):
        offsets = offsets.to(values.device)
        if lengths is None:
            return nested_view_from_values_offsets(values, offsets, ragged_idx)
        return nested_view_from_values_offsets_lengths(values, offsets, lengths.to(values.device), ragged_idx)

    original_offsets = NestedTensor.offsets
    original_sum = NestedTensor.sum

    def offsets(self):
        value = original_offsets(self)
        return value if value.device == self.device else value.to(self.device)

    def sum_(self, *args, **kwargs):
        return self.values().sum() if not args and not kwargs else original_sum(self, *args, **kwargs)

    torch.nested.nested_tensor_from_jagged = nested_tensor_from_jagged
    NestedTensor.offsets = offsets
    NestedTensor.sum = sum_


def check() -> None:
    from verl.utils.torch_functional import expand_as_nested

    device = "cuda" if torch.cuda.is_available() else "cpu"
    nested = torch.nested.as_nested_tensor([torch.tensor([1, 2]), torch.tensor([3])], layout=torch.jagged).to(device)
    temperature = torch.tensor([0.5, 1.5], device=device)
    expanded = expand_as_nested(temperature, nested)
    rebuilt = torch.nested.nested_tensor_from_jagged(expanded.values(), nested.offsets())
    assert torch.equal(expanded.values(), torch.tensor([0.5, 0.5, 1.5], device=device))
    assert rebuilt.values().device.type == device and nested.sum().item() == 6
    import verl.trainer.sft_trainer  # noqa: F401


patch_torch_23()
if sys.argv[1:] == ["--check"]:
    check()
else:
    runpy.run_module("verl.trainer.sft_trainer", run_name="__main__")
