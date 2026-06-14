"""Calibrate thermal diffusivity from sparse noisy sensor observations.

This script demonstrates a digital-twin update step:
  1. Pick one held-out simulation.
  2. Hide its true alpha.
  3. Observe only sparse noisy sensors from the final temperature field.
  4. Grid-search alpha by rerunning the finite-difference simulator.
  5. Choose the alpha with the lowest sensor MSE.

Example:
    python3 src/calibrate.py --sample-idx 0 --n-sensors 16
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

from simulate import simulate_heat


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--sample-idx", type=int, default=0)
    parser.add_argument("--n-sensors", type=int, default=16)
    parser.add_argument("--noise-std", type=float, default=0.01)
    parser.add_argument("--alpha-min", type=float, default=0.02)
    parser.add_argument("--alpha-max", type=float, default=0.10)
    parser.add_argument("--n-candidates", type=int, default=81)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--figure-path", type=Path, default=Path("figures/calibration_curve.png"))
    parser.add_argument("--results-path", type=Path, default=Path("results/calibration_result.json"))
    return parser.parse_args()


def load_metadata(data_dir: Path) -> dict:
    path = data_dir / "metadata.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing metadata file: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_sample(data_dir: Path, split: str, sample_idx: int) -> tuple[np.ndarray, np.ndarray, float]:
    path = data_dir / f"{split}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing data split: {path}")

    data = np.load(path)
    x = data["X"][sample_idx]
    y = data["Y"][sample_idx, 0]
    true_alpha = float(data["alpha"][sample_idx])
    initial = x[0]
    heater = x[1]
    return initial, heater, y, true_alpha


def choose_sensor_locations(
    grid_size: int,
    n_sensors: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Choose random grid locations, avoiding duplicate sensors."""
    n_points = grid_size * grid_size
    if n_sensors > n_points:
        raise ValueError(f"n_sensors={n_sensors} exceeds available grid points={n_points}")

    flat_indices = rng.choice(n_points, size=n_sensors, replace=False)
    rows, cols = np.unravel_index(flat_indices, shape=(grid_size, grid_size))
    return np.stack([rows, cols], axis=1)


def observe_sensors(
    field: np.ndarray,
    sensor_locs: np.ndarray,
    noise_std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    clean = field[sensor_locs[:, 0], sensor_locs[:, 1]]
    noise = rng.normal(0.0, noise_std, size=clean.shape)
    return (clean + noise).astype(np.float32)


def sensor_mse(
    predicted_field: np.ndarray,
    sensor_locs: np.ndarray,
    observed_values: np.ndarray,
) -> float:
    predicted_values = predicted_field[sensor_locs[:, 0], sensor_locs[:, 1]]
    return float(np.mean((predicted_values - observed_values) ** 2))


def grid_search_alpha(
    initial: np.ndarray,
    heater: np.ndarray,
    sensor_locs: np.ndarray,
    observed_values: np.ndarray,
    candidate_alphas: np.ndarray,
    dt: float,
    dx: float,
    steps: int,
) -> np.ndarray:
    losses = np.empty_like(candidate_alphas, dtype=np.float32)
    for i, alpha in enumerate(candidate_alphas):
        predicted = simulate_heat(
            initial=initial,
            source=heater,
            alpha=float(alpha),
            dt=dt,
            dx=dx,
            steps=steps,
        )
        losses[i] = sensor_mse(predicted, sensor_locs, observed_values)
    return losses


def save_calibration_figure(
    candidate_alphas: np.ndarray,
    losses: np.ndarray,
    true_alpha: float,
    recovered_alpha: float,
    figure_path: Path,
) -> None:
    figure_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    ax.plot(candidate_alphas, losses, marker="o", markersize=3, linewidth=1.5)
    ax.axvline(true_alpha, color="black", linestyle="--", linewidth=1.5, label=f"true alpha = {true_alpha:.4f}")
    ax.axvline(
        recovered_alpha,
        color="tab:red",
        linestyle="-",
        linewidth=1.5,
        label=f"recovered alpha = {recovered_alpha:.4f}",
    )
    ax.set_xlabel("Candidate alpha")
    ax.set_ylabel("Sensor MSE")
    ax.set_title("Sparse-sensor calibration of thermal diffusivity")
    ax.legend(frameon=False)
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)


def save_sensor_figure(
    initial: np.ndarray,
    heater: np.ndarray,
    final: np.ndarray,
    sensor_locs: np.ndarray,
    figure_path: Path,
) -> None:
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3), constrained_layout=True)
    panels = [
        (initial, "Initial temperature", "inferno"),
        (heater, "Heater/source", "magma"),
        (final, "Final temperature + sensors", "inferno"),
    ]
    for ax, (field, title, cmap) in zip(axes, panels):
        im = ax.imshow(field, origin="lower", cmap=cmap)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    axes[-1].scatter(
        sensor_locs[:, 1],
        sensor_locs[:, 0],
        s=24,
        facecolors="none",
        edgecolors="cyan",
        linewidths=1.2,
        label="sensors",
    )
    axes[-1].legend(frameon=False, loc="upper right", fontsize=8)
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)


def save_results(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    metadata = load_metadata(args.data_dir)
    initial, heater, final, true_alpha = load_sample(args.data_dir, args.split, args.sample_idx)
    grid_size = initial.shape[0]

    sensor_locs = choose_sensor_locations(grid_size, args.n_sensors, rng)
    observed_values = observe_sensors(final, sensor_locs, args.noise_std, rng)
    candidate_alphas = np.linspace(args.alpha_min, args.alpha_max, args.n_candidates, dtype=np.float32)

    losses = grid_search_alpha(
        initial=initial,
        heater=heater,
        sensor_locs=sensor_locs,
        observed_values=observed_values,
        candidate_alphas=candidate_alphas,
        dt=float(metadata["dt"]),
        dx=float(metadata["dx"]),
        steps=int(metadata["steps"]),
    )

    best_idx = int(np.argmin(losses))
    recovered_alpha = float(candidate_alphas[best_idx])
    best_sensor_mse = float(losses[best_idx])

    save_calibration_figure(
        candidate_alphas=candidate_alphas,
        losses=losses,
        true_alpha=true_alpha,
        recovered_alpha=recovered_alpha,
        figure_path=args.figure_path,
    )
    sensor_figure_path = args.figure_path.with_name("calibration_sensors.png")
    save_sensor_figure(initial, heater, final, sensor_locs, sensor_figure_path)

    results = {
        "split": args.split,
        "sample_idx": args.sample_idx,
        "true_alpha": true_alpha,
        "recovered_alpha": recovered_alpha,
        "absolute_error": abs(recovered_alpha - true_alpha),
        "best_sensor_mse": best_sensor_mse,
        "n_sensors": args.n_sensors,
        "noise_std": args.noise_std,
        "candidate_alpha_min": args.alpha_min,
        "candidate_alpha_max": args.alpha_max,
        "n_candidates": args.n_candidates,
        "sensor_locations": sensor_locs.tolist(),
        "figure_path": str(args.figure_path),
        "sensor_figure_path": str(sensor_figure_path),
    }
    save_results(args.results_path, results)

    print(f"true alpha:      {true_alpha:.6f}")
    print(f"recovered alpha: {recovered_alpha:.6f}")
    print(f"absolute error:  {abs(recovered_alpha - true_alpha):.6f}")
    print(f"sensor MSE:      {best_sensor_mse:.6e}")
    print(f"Saved curve:     {args.figure_path}")
    print(f"Saved sensors:   {sensor_figure_path}")
    print(f"Saved results:   {args.results_path}")


if __name__ == "__main__":
    main()
