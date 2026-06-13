"""Generate synthetic data for a 2D heat-equation surrogate model.

Each sample contains:
  X[0] = initial temperature field
  X[1] = fixed heater/source field
  X[2] = thermal diffusivity alpha repeated on the grid
  Y[0] = final temperature field after finite-difference simulation

Example:
    python src/simulate.py --n-train 1000 --n-val 200 --n-test 200 --n-ood 200
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
from tqdm import tqdm


def gaussian_field(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    centers: list[tuple[float, float]],
    amplitudes: list[float],
    sigmas: list[float],
) -> np.ndarray:
    """Build a smooth field from a small number of Gaussian bumps."""
    field = np.zeros_like(grid_x, dtype=np.float32)
    for (cx, cy), amp, sigma in zip(centers, amplitudes, sigmas):
        radius_sq = (grid_x - cx) ** 2 + (grid_y - cy) ** 2
        field += amp * np.exp(-radius_sq / (2.0 * sigma**2))
    return field.astype(np.float32)


def make_initial_condition(
    rng: np.random.Generator,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
) -> np.ndarray:
    """Random smooth initial temperature, kept mild so the heater is visible."""
    n_bumps = rng.integers(1, 4)
    centers = [(rng.uniform(0.2, 0.8), rng.uniform(0.2, 0.8)) for _ in range(n_bumps)]
    amplitudes = rng.uniform(0.1, 0.8, size=n_bumps).tolist()
    sigmas = rng.uniform(0.05, 0.16, size=n_bumps).tolist()
    field = gaussian_field(grid_x, grid_y, centers, amplitudes, sigmas)
    field += rng.normal(0.0, 0.01, size=field.shape).astype(np.float32)
    field = np.clip(field, 0.0, None)
    return field.astype(np.float32)


def make_heater_field(
    rng: np.random.Generator,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    mode: str,
) -> np.ndarray:
    """Create in-distribution or OOD heater/source patterns."""
    if mode == "center":
        n_heaters = 1
        centers = [(rng.uniform(0.35, 0.65), rng.uniform(0.35, 0.65))]
    elif mode == "corner":
        n_heaters = 1
        corner_centers = [(0.15, 0.15), (0.15, 0.85), (0.85, 0.15), (0.85, 0.85)]
        cx, cy = corner_centers[rng.integers(0, len(corner_centers))]
        centers = [(rng.normal(cx, 0.03), rng.normal(cy, 0.03))]
    elif mode == "multi":
        n_heaters = rng.integers(2, 5)
        centers = [(rng.uniform(0.12, 0.88), rng.uniform(0.12, 0.88)) for _ in range(n_heaters)]
    else:
        raise ValueError(f"Unknown heater mode: {mode}")

    amplitudes = rng.uniform(2.0, 6.0, size=n_heaters).tolist()
    sigmas = rng.uniform(0.035, 0.08, size=n_heaters).tolist()
    return gaussian_field(grid_x, grid_y, centers, amplitudes, sigmas)


def laplacian_neumann(u: np.ndarray, dx: float) -> np.ndarray:
    """Second-order 5-point Laplacian with zero-flux boundaries."""
    padded = np.pad(u, pad_width=1, mode="edge")
    lap = (
        padded[1:-1, 2:]
        + padded[1:-1, :-2]
        + padded[2:, 1:-1]
        + padded[:-2, 1:-1]
        - 4.0 * u
    ) / dx**2
    return lap.astype(np.float32)


def simulate_heat(
    initial: np.ndarray,
    source: np.ndarray,
    alpha: float,
    dt: float,
    dx: float,
    steps: int,
) -> np.ndarray:
    """Explicit finite-difference solve for du/dt = alpha * lap(u) + source."""
    u = initial.astype(np.float32).copy()
    source = source.astype(np.float32)
    for _ in range(steps):
        u += dt * (alpha * laplacian_neumann(u, dx) + source)
    return u.astype(np.float32)


def make_sample(
    rng: np.random.Generator,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    alpha_range: tuple[float, float],
    heater_mode: str,
    dt: float,
    dx: float,
    steps: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    initial = make_initial_condition(rng, grid_x, grid_y)
    heater = make_heater_field(rng, grid_x, grid_y, heater_mode)
    alpha = float(rng.uniform(*alpha_range))
    final = simulate_heat(initial, heater, alpha, dt, dx, steps)

    alpha_channel = np.full_like(initial, alpha, dtype=np.float32)
    x = np.stack([initial, heater, alpha_channel], axis=0).astype(np.float32)
    y = final[None, :, :].astype(np.float32)
    return x, y, alpha


def generate_split(
    name: str,
    n_samples: int,
    rng: np.random.Generator,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    alpha_range: tuple[float, float],
    heater_mode: str,
    dt: float,
    dx: float,
    steps: int,
    out_dir: Path,
) -> None:
    xs = np.empty((n_samples, 3, grid_x.shape[0], grid_x.shape[1]), dtype=np.float32)
    ys = np.empty((n_samples, 1, grid_x.shape[0], grid_x.shape[1]), dtype=np.float32)
    alphas = np.empty((n_samples,), dtype=np.float32)

    for i in tqdm(range(n_samples), desc=f"Generating {name}", leave=False):
        xs[i], ys[i], alphas[i] = make_sample(
            rng=rng,
            grid_x=grid_x,
            grid_y=grid_y,
            alpha_range=alpha_range,
            heater_mode=heater_mode,
            dt=dt,
            dx=dx,
            steps=steps,
        )

    out_path = out_dir / f"{name}.npz"
    np.savez_compressed(
        out_path,
        X=xs,
        Y=ys,
        alpha=alphas,
        heater_mode=np.array(heater_mode),
    )
    print(f"Saved {out_path} | X {xs.shape} | Y {ys.shape}")


def save_preview(data_path: Path, figure_path: Path) -> None:
    data = np.load(data_path)
    x = data["X"][0]
    y = data["Y"][0, 0]

    fig, axes = plt.subplots(1, 3, figsize=(10, 3), constrained_layout=True)
    panels = [
        (x[0], "Initial temperature"),
        (x[1], "Heater/source"),
        (y, "Final temperature"),
    ]
    for ax, (field, title) in zip(axes, panels):
        im = ax.imshow(field, origin="lower", cmap="inferno")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)
    print(f"Saved preview figure to {figure_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("data"))
    parser.add_argument("--figure-dir", type=Path, default=Path("figures"))
    parser.add_argument("--grid-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--dt", type=float, default=1.0e-3)
    parser.add_argument("--alpha-min", type=float, default=0.02)
    parser.add_argument("--alpha-max", type=float, default=0.08)
    parser.add_argument("--n-train", type=int, default=1000)
    parser.add_argument("--n-val", type=int, default=200)
    parser.add_argument("--n-test", type=int, default=200)
    parser.add_argument("--n-ood", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)

    alpha_range = (args.alpha_min, args.alpha_max)
    dx = 1.0 / (args.grid_size - 1)
    stable_dt = dx**2 / (4.0 * args.alpha_max)
    if args.dt > stable_dt:
        raise ValueError(
            f"dt={args.dt} is too large for explicit diffusion stability. "
            f"Use dt <= {stable_dt:.4e} for alpha_max={args.alpha_max}."
        )

    coords = np.linspace(0.0, 1.0, args.grid_size, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(coords, coords, indexing="ij")
    rng = np.random.default_rng(args.seed)

    config = vars(args).copy()
    config["out_dir"] = str(args.out_dir)
    config["figure_dir"] = str(args.figure_dir)
    config["dx"] = dx
    config["equation"] = "du/dt = alpha * laplacian(u) + q(x,y)"
    config["boundary_condition"] = "zero-flux Neumann"
    with (args.out_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    split_specs = [
        ("train", args.n_train, "center"),
        ("val", args.n_val, "center"),
        ("test", args.n_test, "center"),
        ("ood_corner", args.n_ood, "corner"),
        ("ood_multi", args.n_ood, "multi"),
    ]
    for name, n_samples, heater_mode in split_specs:
        if n_samples <= 0:
            continue
        generate_split(
            name=name,
            n_samples=n_samples,
            rng=rng,
            grid_x=grid_x,
            grid_y=grid_y,
            alpha_range=alpha_range,
            heater_mode=heater_mode,
            dt=args.dt,
            dx=dx,
            steps=args.steps,
            out_dir=args.out_dir,
        )

    train_path = args.out_dir / "train.npz"
    if train_path.exists():
        save_preview(train_path, args.figure_dir / "simulation_example.png")


if __name__ == "__main__":
    main()
