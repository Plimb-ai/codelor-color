"""Shared pytest fixtures."""

from __future__ import annotations

import pytest
import torch


@pytest.fixture()
def single_thread():
    """Force single-threaded torch so floating-point results are reproducible.

    Threaded BLAS is not bit-deterministic on every platform, which would make
    cross-run equality assertions flaky.
    """
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)
