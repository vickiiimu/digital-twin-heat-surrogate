"""Estimate predictive uncertainty with an ensemble of trained CNN models.

Train an ensemble first, for example:
    python3 src/train.py --epochs 30 --batch_size 32 --seed 0 --run-name cnn_ensemble_seed0
    python3 src/train.py --epochs 30 --batch_size 32 --seed 1 --run-name cnn_ensemble_seed1
    python3 src/train.py --epochs 30 --batch_size 32 --seed 2 --run-name cnn_ensemble_seed2

Then run:
    python3 src/uncertainty.py

This saves:
    figures/uncertainty_map.png
    results/uncertainty_result.json
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset import HeatDataset, NormalizationStats, load_normalization_stats
from models import HeatCNN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument(
        "--run-names",
        nargs="+",
        default=["cnn_ensemble_seed0", "cnn_ensemble_seed1", "cnn_ensemble_seed2"],
        help="Run names whose best checkpoints should be loaded.",
    )
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        type=Path,
        default=None,
        help="Explicit checkpoint paths. Overrides --run-names.",
    )
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--sample-idx", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--figure-path", type=Path, default=Path("figures/uncertainty_map.png"))
    parser.add_argument("--results-path", type=Path, default=Path("results/uncertainty_result.json"))
    return parser.parse_args()


def get_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def checkpoint_paths(args: argparse.Namespace) -> list[Path]:
    if args.checkpoints is not None:
        paths = args.checkpoints
    else:
        paths = [args.checkpoint_dir / f"{run_name}_best.pt" for run_name in args.run_names]

    missing = [path for path in paths if not path.exists()]
    if missing:
        formatted = "\n".join(f"  {path}" for path in missing)
        raise FileNotFoundError(
            "Missing ensemble checkpoint(s):\n"
            f"{formatted}\n\n"
            "Train them first, for example:\n"
            "  python3 src/train.py --epochs 30 --batch_size 32 --seed 0 --run-name cnn_ensemble_seed0\n"
            "  python3 src/train.py --epochs 30 --batch_size 32 --seed 1 --run-name cnn_ensemble_seed1\n"
            "  python3 src/train.py --epochs 30 --batch_size 32 --seed 2 --run-name cnn_ensemble_seed2"
        )
    return paths


def stats_from_checkpoint(ckpt: dict, fallback: NormalizationStats) -> NormalizationStats:
    payload = ckpt.get("normalization_stats")
    if payload is None:
        return fallback
    return NormalizationStats(
        x_mean=payload["x_mean"].float(),
        x_std=payload["x_std"].float(),
        y_mean=payload["y_mean"].float(),
        y_std=payload["y_std"].float(),
    )


def load_model(path: Path, device: torch.device) -> tuple[HeatCNN, NormalizationStats | None, dict]:
    ckpt = torch.load(path, map_location=device)
    ckpt_args = ckpt.get("args", {})
    model = HeatCNN(
        hidden_channels=int(ckpt_args.get("hidden_channels", 64)),
        depth=int(ckpt_args.get("depth", 5)),
        residual_initial=not bool(ckpt_args.get("no_residual_initial", False)),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt.get("normalization_stats"), ckpt_args


def denormalize_y(y: torch.Tensor, stats: NormalizationStats) -> torch.Tensor:
    return y * stats.y_std.to(y.device) + stats.y_mean.to(y.device)


@torch.no_grad()
def ensemble_predict(
    models: list[HeatCNN],
    x: torch.Tensor,
    stats: NormalizationStats,
) -> torch.Tensor:
    preds = []
    for model in models:
        pred_norm = model(x)
        pred_phys = denormalize_y(pred_norm, stats)
        preds.append(pred_phys.cpu())
    return torch.stack(preds, dim=0)


def relative_l2(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.linalg.norm((pred - target).ravel()) / (np.linalg.norm(target.ravel()) + 1.0e-8))


def save_uncertainty_figure(
    initial: np.ndarray,
    heater: np.ndarray,
    target: np.ndarray,
    pred_mean: np.ndarray,
    pred_std: np.ndarray,
    alpha: float,
    rel_l2: float,
    figure_path: Path,
) -> None:
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    error = np.abs(pred_mean - target)

    fig, axes = plt.subplots(1, 6, figsize=(16, 3), constrained_layout=True)
    panels = [
        (initial, "Initial", "inferno"),
        (heater, "Heater", "magma"),
        (target, "Ground truth", "inferno"),
        (pred_mean, "Ensemble mean", "inferno"),
        (error, "Absolute error", "viridis"),
        (pred_std, "Ensemble std", "viridis"),
    ]
    for ax, (field, title, cmap) in zip(axes, panels):
        im = ax.imshow(field, origin="lower", cmap=cmap)
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"Ensemble uncertainty | alpha = {alpha:.4f} | rel L2 = {rel_l2:.4f}", y=1.05)
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)


def save_results(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    paths = checkpoint_paths(args)

    fallback_stats = load_normalization_stats(args.data_dir)
    first_ckpt = torch.load(paths[0], map_location="cpu")
    stats = stats_from_checkpoint(first_ckpt, fallback_stats)
    stats_device = NormalizationStats(
        x_mean=stats.x_mean.to(device),
        x_std=stats.x_std.to(device),
        y_mean=stats.y_mean.to(device),
        y_std=stats.y_std.to(device),
    )

    dataset = HeatDataset(args.data_dir / f"{args.split}.npz", stats=stats, normalize=True)
    sample = dataset[args.sample_idx]
    x = sample["x"].unsqueeze(0).to(device)
    target = dataset.y[args.sample_idx, 0].numpy()
    initial = dataset.x[args.sample_idx, 0].numpy()
    heater = dataset.x[args.sample_idx, 1].numpy()
    alpha = float(dataset.alpha[args.sample_idx])

    models = []
    model_args = []
    for path in paths:
        model, _, ckpt_args = load_model(path, device)
        models.append(model)
        model_args.append(ckpt_args)

    preds = ensemble_predict(models, x, stats_device).squeeze(2).numpy()
    pred_mean = preds.mean(axis=0)[0]
    pred_std = preds.std(axis=0)[0]
    rel_error = relative_l2(pred_mean, target)
    mse = float(np.mean((pred_mean - target) ** 2))
    mean_uncertainty = float(pred_std.mean())
    max_uncertainty = float(pred_std.max())

    save_uncertainty_figure(
        initial=initial,
        heater=heater,
        target=target,
        pred_mean=pred_mean,
        pred_std=pred_std,
        alpha=alpha,
        rel_l2=rel_error,
        figure_path=args.figure_path,
    )

    results = {
        "split": args.split,
        "sample_idx": args.sample_idx,
        "alpha": alpha,
        "checkpoints": [str(path) for path in paths],
        "n_models": len(models),
        "mse": mse,
        "rel_l2": rel_error,
        "mean_uncertainty": mean_uncertainty,
        "max_uncertainty": max_uncertainty,
        "figure_path": str(args.figure_path),
    }
    save_results(args.results_path, results)

    print(f"Loaded {len(models)} model(s) on {device}")
    print(f"Sample: {args.split}[{args.sample_idx}]")
    print(f"alpha: {alpha:.6f}")
    print(f"ensemble MSE: {mse:.6e}")
    print(f"ensemble rel L2: {rel_error:.6f}")
    print(f"mean uncertainty: {mean_uncertainty:.6e}")
    print(f"max uncertainty: {max_uncertainty:.6e}")
    print(f"Saved figure: {args.figure_path}")
    print(f"Saved results: {args.results_path}")


if __name__ == "__main__":
    main()
