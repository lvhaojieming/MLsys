#!/usr/bin/env python3
"""Repository entry point for MoQE L1 Router training."""

from __future__ import annotations

import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from moqe_router.training.trainer import main  # noqa: E402


if __name__ == "__main__":
    main()
