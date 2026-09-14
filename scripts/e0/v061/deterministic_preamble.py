"""Deterministic execution preamble — V06-PREEXEC-CLOSURE-CORRIGENDUM-1.

Every smoke, replay, and evidence training process MUST import and call
`apply()` BEFORE any CUDA model work. The function establishes the frozen
deterministic configuration and then queries the post-application values.

The frozen configuration (per Q2_Q3_EXECUTION_RELEASE.md SS18):
  - CUBLAS_WORKSPACE_CONFIG=:4096:8 (set before CUDA context creation)
  - torch.use_deterministic_algorithms(True)
  - torch.backends.cuda.matmul.allow_tf32 = False
  - torch.backends.cudnn.allow_tf32 = False
  - torch.backends.cudnn.deterministic = True
  - torch.backends.cudnn.benchmark = False
"""

from __future__ import annotations

import os


def apply() -> dict:
    """Apply the frozen deterministic configuration and return the
    post-application queried values (not process defaults)."""
    # CUBLAS_WORKSPACE_CONFIG must be set before CUDA context creation
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    import torch

    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    return query()


def query() -> dict:
    """Query the current deterministic state (post-application check)."""
    import torch

    return {
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "are_deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
    }


def verify_applied(state: dict) -> bool:
    """Verify that all post-application values match the frozen contract."""
    return (
        state["cublas_workspace_config"] == ":4096:8"
        and state["are_deterministic_algorithms_enabled"] is True
        and state["cuda_matmul_allow_tf32"] is False
        and state["cudnn_allow_tf32"] is False
        and state["cudnn_deterministic"] is True
        and state["cudnn_benchmark"] is False
    )
