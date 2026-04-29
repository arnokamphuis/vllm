# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import types

import pytest

from tests.utils import spawn_new_process_for_each_test


@spawn_new_process_for_each_test
@pytest.mark.parametrize(
    "device_capability,has_rmsnorm,has_fused_add_rms_norm,expected_available,expected_fused",
    [
        # Case 1: CUDA not available
        (None, False, False, False, False),
        # Case 2: CUDA available but < SM100
        ((9, 0), True, False, False, False),
        # Case 3: CUDA available and SM100, rmsnorm op registered
        ((10, 0), True, False, True, False),
        # Case 4: SM100 with both rmsnorm and fused_add_rms_norm
        ((10, 0), True, True, True, True),
    ],
)
def test_oink_availability_checks(
    device_capability: tuple[int, int] | None,
    has_rmsnorm: bool,
    has_fused_add_rms_norm: bool,
    expected_available: bool,
    expected_fused: bool,
    monkeypatch,
):
    """Test OINK support detection with clean import state for each parameter set."""
    # Import locally to avoid spawn issue
    import torch

    # Mock torch.cuda before any imports of vllm
    cuda_available = device_capability is not None
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda_available)
    if device_capability is not None:
        monkeypatch.setattr(
            torch.cuda, "get_device_capability", lambda idx: device_capability
        )

    # Create mock oink ops namespace
    oink_ops = types.SimpleNamespace()
    if has_rmsnorm:
        oink_ops.rmsnorm = lambda x, w, eps: x
    if has_fused_add_rms_norm:
        oink_ops.fused_add_rms_norm = lambda x, residual, w, eps: None

    monkeypatch.setattr(torch.ops, "oink", oink_ops, raising=False)

    # Now import vllm.kernels.oink_ops and IR ops
    import vllm.kernels.oink_ops  # noqa: F401
    from vllm.ir.ops import fused_add_rms_norm, rms_norm

    # Verify support checks
    assert rms_norm.impls["oink"].supported is expected_available
    assert fused_add_rms_norm.impls["oink"].supported is expected_fused


def test_can_view_as_2d_stride_guard():
    # No global import
    import torch

    # Import the helper from the kernels module.
    from vllm.kernels.oink_ops import _can_view_as_2d

    x = torch.zeros((2, 3, 4))
    assert _can_view_as_2d(x) is True

    # Size-1 dims should be ignored by the viewability check.
    # Create a tensor where stride(0) != stride(1) * size(1) due to padding,
    # but view(-1, H) is still valid because dim 1 has size 1.
    base = torch.zeros((2, 10, 4))
    x_singleton = base[:, :1, :]
    x_singleton.view(-1, x_singleton.shape[-1])
    assert _can_view_as_2d(x_singleton) is True

    # Middle-dimension stride break: view(-1, hidden) should be invalid.
    x2 = x[:, ::2, :]
    with pytest.raises(RuntimeError):
        x2.view(-1, x2.shape[-1])
    assert _can_view_as_2d(x2) is False
