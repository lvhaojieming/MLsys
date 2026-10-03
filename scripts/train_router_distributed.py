#!/usr/bin/env python3
"""Run under torchrun; one process per accelerator."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from moqe_router.training.distributed import main

if __name__ == "__main__":
    main()
