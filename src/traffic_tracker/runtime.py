from __future__ import annotations

import os


def configure_cpu_runtime(threads: int) -> None:
    """Set conservative CPU thread settings before detector initialization."""
    threads = max(1, int(threads))
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)

    import torch

    if hasattr(torch.backends, "nnpack"):
        torch.backends.nnpack.set_flags(False)

    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(max(1, min(4, threads)))
    except RuntimeError:
        pass
