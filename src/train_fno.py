"""Convenience entry point for training the Fourier Neural Operator.

This wrapper forwards to `src/train.py` with `--model fno` inserted, so the FNO
uses the same logging, metrics, checkpoints, and OOD evaluation as the CNN.

Local CPU example, safest for Apple Silicon if FFTs fail on MPS:
    python3 src/train_fno.py --epochs 50 --batch-size 32 --device cpu --run-name fno_local_50ep

CUDA cluster example:
    python3 src/train_fno.py --epochs 100 --batch-size 64 --device cuda --num-workers 4 --run-name fno_cuda_100ep
"""

from __future__ import annotations

import sys

from train import main


if __name__ == "__main__":
    if "--model" not in sys.argv:
        sys.argv[1:1] = ["--model", "fno"]
    main()
