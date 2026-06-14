"""Compare finite-difference and surrogate-based alpha calibration.

Both methods use the same sparse noisy sensor observations from one held-out
final temperature field. The finite-difference method reruns the simulator for
each candidate alpha. The surrogate method changes only the alpha input channel
and evaluates a trained neural model for each candidate alpha.

Example:
    python3 src/surrogate_calibrate.py \
      --checkpoint checkpoints/cnn_local_baseline_50ep_best.pt \
      --sample-idx 0 \
      --n-sensors 16
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

from calibrate import (
    choose_sensor_locations,
    grid_search_alpha,
    load_metadata,
    load_sample,
    observe_sensors,
    sensor_mse,
)
from dataset import NormalizationStats, load_normalization_stats
from models import HeatCNN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/cnn_local_baseline_50ep_best.pt"))
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--sample-idx", type=int, default=0)
    parser.add_argument("--n-sensors", type=int, default=16)
    parser.add_argument("--noise-std", type=float, default=0.01)
    parser.add_argument("--alpha-min", type=float, default=0.02)
    parser.add_argument("--alpha-max", type=float, default=0.10)
    parser.add_argument("--n-candidates", type=int, default=81)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--figure-path", type=Path, default=Path("figures/surrogate_calibration_comparison.png"))
    parser.add_argument("--results-path", type=Path, default=Path("results/surrogate_calibration_comparison.json"))
    return parser.parse_args()


def get_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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


def load_surrogate(checkpoint: Path, device: torch.device, data_dir: Path) -> tuple[HeatCNN, NormalizationStats, dict]:
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Missing checkpoint: {checkpoint}\n"
            "Train a CNN first, or pass --checkpoint checkpoints/<run_name>_best.pt"
        )

    ckpt = torch.load(checkpoint, map_location=device)
    ckpt_args = ckpt.get("args", {})
    model = HeatCNN(
        hidden_channels=int(ckpt_args.get("hidden_channels", 64)),
        depth=int(ckpt_args.get("depth", 5)),
        residual_initial=not bool(ckpt_args.get("no_residual_initial", False)),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    fallback_stats = load_normalization_stats(data_dir)
    stats = stats_from_checkpoint(ckpt, fallback_stats)
    stats = NormalizationStats(
        x_mean=stats.x_mean.to(device),
        x_std=stats.x_std.to(device),
        y_mean=stats.y_mean.to(device),
        y_std=stats.y_std.to(device),
    )
    return model, stats, ckpt_args


def make_surrogate_input(
    initial: np.ndarray,
    heater: np.ndarray,
    alpha: float,
    stats: NormalizationStats,
    device: torch.device,
) -> torch.Tensor:
    alpha_channel = np.full_like(initial, alpha, dtype=np.float32)
    x = np.stack([initial, heater, alpha_channel], axis=0)[None]
    x_tensor = torch.from_numpy(x).float().to(device)
    return (x_tensor - stats.x_mean) / stats.x_std


def denormalize_prediction(y_norm: torch.Tensor, stats: NormalizationStats) -> torch.Tensor:
    return y_norm * stats.y_std + stats.y_mean


@torch.no_grad()
def grid_search_surrogate_alpha(
    model: HeatCNN,
    stats: NormalizationStats,
    initial: np.ndarray,
    heater: np.ndarray,
    sensor_locs: np.ndarray,
    observed_values: np.ndarray,
    candidate_alphas: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    losses = np.empty_like(candidate_alphas, dtype=np.float32)
    for i, alpha in enumerate(candidate_alphas):
        x = make_surrogate_input(initial, heater, float(alpha), stats, device)
        pred_norm = model(x)
        pred = denormalize_prediction(pred_norm, stats).squeeze().cpu().numpy()
        losses[i] = sensor_mse(pred, sensor_locs, observed_values)
    return losses


def save_comparison_figure(
    candidate_alphas: np.ndarray,
    fd_losses: np.ndarray,
    surrogate_losses: np.ndarray,
    true_alpha: float,
    fd_alpha: float,
    surrogate_alpha: float,
    figure_path: Path,
) -> None:
    figure_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    ax.plot(candidate_alphas, fd_losses, marker="o", markersize=3, linewidth=1.5, label="finite difference")
    ax.plot(candidate_alphas, surrogate_losses, marker="s", markersize=3, linewidth=1.5, label="CNN surrogate")
    ax.axvline(true_alpha, color="black", linestyle="--", linewidth=1.5, label=f"true alpha = {true_alpha:.4f}")
    ax.axvline(fd_alpha, color="tab:blue", linestyle=":", linewidth=2.0, label=f"FD recovered = {fd_alpha:.4f}")
    ax.axvline(
        surrogate_alpha,
        color="tab:orange",
        linestyle=":",
        linewidth=2.0,
        label=f"surrogate recovered = {surrogate_alpha:.4f}",
    )
    ax.set_xlabel("Candidate alpha")
    ax.set_ylabel("Sensor MSE")
    ax.set_title("Finite-difference vs surrogate sparse-sensor calibration")
    ax.legend(frameon=False)
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)


def save_results(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    device = get_device(args.device)

    metadata = load_metadata(args.data_dir)
    initial, heater, final, true_alpha = load_sample(args.data_dir, args.split, args.sample_idx)
    sensor_locs = choose_sensor_locations(initial.shape[0], args.n_sensors, rng)
    observed_values = observe_sensors(final, sensor_locs, args.noise_std, rng)
    candidate_alphas = np.linspace(args.alpha_min, args.alpha_max, args.n_candidates, dtype=np.float32)

    fd_losses = grid_search_alpha(
        initial=initial,
        heater=heater,
        sensor_locs=sensor_locs,
        observed_values=observed_values,
        candidate_alphas=candidate_alphas,
        dt=float(metadata["dt"]),
        dx=float(metadata["dx"]),
        steps=int(metadata["steps"]),
    )

    model, stats, ckpt_args = load_surrogate(args.checkpoint, device, args.data_dir)
    surrogate_losses = grid_search_surrogate_alpha(
        model=model,
        stats=stats,
        initial=initial,
        heater=heater,
        sensor_locs=sensor_locs,
        observed_values=observed_values,
        candidate_alphas=candidate_alphas,
        device=device,
    )

    fd_idx = int(np.argmin(fd_losses))
    surrogate_idx = int(np.argmin(surrogate_losses))
    fd_alpha = float(candidate_alphas[fd_idx])
    surrogate_alpha = float(candidate_alphas[surrogate_idx])

    save_comparison_figure(
        candidate_alphas=candidate_alphas,
        fd_losses=fd_losses,
        surrogate_losses=surrogate_losses,
        true_alpha=true_alpha,
        fd_alpha=fd_alpha,
        surrogate_alpha=surrogate_alpha,
        figure_path=args.figure_path,
    )

    results = {
        "split": args.split,
        "sample_idx": args.sample_idx,
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "true_alpha": true_alpha,
        "finite_difference_recovered_alpha": fd_alpha,
        "surrogate_recovered_alpha": surrogate_alpha,
        "finite_difference_abs_error": abs(fd_alpha - true_alpha),
        "surrogate_abs_error": abs(surrogate_alpha - true_alpha),
        "finite_difference_best_sensor_mse": float(fd_losses[fd_idx]),
        "surrogate_best_sensor_mse": float(surrogate_losses[surrogate_idx]),
        "n_sensors": args.n_sensors,
        "noise_std": args.noise_std,
        "candidate_alpha_min": args.alpha_min,
        "candidate_alpha_max": args.alpha_max,
        "n_candidates": args.n_candidates,
        "sensor_locations": sensor_locs.tolist(),
        "figure_path": str(args.figure_path),
        "model_args": ckpt_args,
    }
    save_results(args.results_path, results)

    print(f"true alpha:                 {true_alpha:.6f}")
    print(f"finite-difference alpha:    {fd_alpha:.6f}")
    print(f"surrogate alpha:            {surrogate_alpha:.6f}")
    print(f"finite-difference abs err:  {abs(fd_alpha - true_alpha):.6f}")
    print(f"surrogate abs err:          {abs(surrogate_alpha - true_alpha):.6f}")
    print(f"finite-difference best MSE: {float(fd_losses[fd_idx]):.6e}")
    print(f"surrogate best MSE:         {float(surrogate_losses[surrogate_idx]):.6e}")
    print(f"Saved figure:               {args.figure_path}")
    print(f"Saved results:              {args.results_path}")


if __name__ == "__main__":
    main()
