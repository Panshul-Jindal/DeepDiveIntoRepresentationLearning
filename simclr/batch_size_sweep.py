"""
SimCLR Batch Size Sweep Entry Point.
Runs batch sizes 128, 256, 512 and training time sweep (100, 125, 150, 175, 200 epochs).
"""

from __future__ import annotations
import sys
from pathlib import Path

_SIMCLR_DIR = Path(__file__).resolve().parent
if str(_SIMCLR_DIR) not in sys.path:
    sys.path.insert(0, str(_SIMCLR_DIR))

from batch_size_training_time import main

if __name__ == "__main__":
    main()

