"""Train surrogate models for the synthetic 2D heat equation dataset.

CNN local example:
    python3 src/train.py --model cnn --epochs 50 --batch-size 32

FNO local example:
    python3 src/train.py --model fno --epochs 50 --batch-size 32 --device cpu

Cluster example:
    python3 src/train.py --model fno --epochs 100 --batch-size 64 --num-workers 4 --device cuda
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from dataset import NormalizationStats, make_dataloaders
from models import HeatCNN, HeatFNO, count_parameters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["cnn", "fno"], default="cnn")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--log-dir", type=Path, default=Path("logs"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--run-name", type=str, default=None)

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--fno-width", type=int, default=32)
    parser.add_argument("--fno-modes", type=int, default=12)
    parser.add_argument("--fno-depth", type=int, default=4)
    parser.add_argument("--fno-no-grid", action="store_true")
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--no-residual-initial", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_model(args: argparse.Namespace) -> nn.Module:
    if args.model == "cnn":
        return HeatCNN(
            hidden_channels=args.hidden_channels,
            depth=args.depth,
            residual_initial=not args.no_residual_initial,
        )
    if args.model == "fno":
        return HeatFNO(
            width=args.fno_width,
            modes1=args.fno_modes,
            modes2=args.fno_modes,
            depth=args.fno_depth,
            use_grid=not args.fno_no_grid,
            residual_initial=not args.no_residual_initial,
        )
    raise ValueError(f"Unknown model: {args.model}")


def stats_to_device(stats: NormalizationStats | None, device: torch.device) -> NormalizationStats | None:
    if stats is None:
        return None
    return NormalizationStats(
        x_mean=stats.x_mean.to(device),
        x_std=stats.x_std.to(device),
        y_mean=stats.y_mean.to(device),
        y_std=stats.y_std.to(device),
    )


def denormalize_y(y: torch.Tensor, stats: NormalizationStats | None) -> torch.Tensor:
    if stats is None:
        return y
    return y * stats.y_std.to(y.device) + stats.y_mean.to(y.device)


def compute_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    stats: NormalizationStats | None,
) -> dict[str, float]:
    """Compute metrics in physical temperature units."""
    pred_phys = denormalize_y(pred, stats)
    target_phys = denormalize_y(target, stats)
    diff = pred_phys - target_phys

    mse = torch.mean(diff**2)
    rel_l2 = torch.linalg.vector_norm(diff.flatten(1), dim=1) / (
        torch.linalg.vector_norm(target_phys.flatten(1), dim=1) + 1.0e-8
    )
    max_abs = torch.amax(torch.abs(diff), dim=(1, 2, 3))
    return {
        "mse": float(mse.detach().cpu()),
        "rel_l2": float(rel_l2.mean().detach().cpu()),
        "max_abs": float(max_abs.mean().detach().cpu()),
    }


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0

    for batch in tqdm(loader, desc="train", leave=False):
        x = batch["x"].to(device)
        y = batch["y"].to(device)

        optimizer.zero_grad(set_to_none=True)
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()

        batch_size = x.shape[0]
        total_loss += float(loss.detach().cpu()) * batch_size
        total_samples += batch_size

    return total_loss / total_samples


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    stats: NormalizationStats | None,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    metric_sums = {"mse": 0.0, "rel_l2": 0.0, "max_abs": 0.0}

    for batch in tqdm(loader, desc="eval", leave=False):
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        pred = model(x)
        loss = criterion(pred, y)
        metrics = compute_metrics(pred, y, stats)

        batch_size = x.shape[0]
        total_loss += float(loss.detach().cpu()) * batch_size
        total_samples += batch_size
        for key, value in metrics.items():
            metric_sums[key] += value * batch_size

    return {
        "loss": total_loss / total_samples,
        "mse": metric_sums["mse"] / total_samples,
        "rel_l2": metric_sums["rel_l2"] / total_samples,
        "max_abs": metric_sums["max_abs"] / total_samples,
    }


def write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def serializable_args(args: argparse.Namespace) -> dict:
    payload = vars(args).copy()
    for key, value in payload.items():
        if isinstance(value, Path):
            payload[key] = str(value)
    return payload


def append_history(path: Path, row: dict[str, float | int]) -> None:
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def checkpoint_payload(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    args: argparse.Namespace,
    stats: NormalizationStats | None,
    metrics: dict[str, float],
) -> dict:
    stats_payload = None
    if stats is not None:
        stats_payload = {
            "x_mean": stats.x_mean.cpu(),
            "x_std": stats.x_std.cpu(),
            "y_mean": stats.y_mean.cpu(),
            "y_std": stats.y_std.cpu(),
        }
    return {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "args": serializable_args(args),
        "normalization_stats": stats_payload,
        "metrics": metrics,
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"{args.model}_{timestamp}"
    run_dir = args.log_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    loaders, stats = make_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        normalize=not args.no_normalize,
        include_ood=True,
    )
    stats_device = stats_to_device(stats, device)

    model = make_model(args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()

    config = {
        **serializable_args(args),
        "run_name": run_name,
        "device": str(device),
        "parameter_count": count_parameters(model),
        "normalization": None
        if stats is None
        else {
            "x_mean": stats.x_mean.flatten().tolist(),
            "x_std": stats.x_std.flatten().tolist(),
            "y_mean": stats.y_mean.flatten().tolist(),
            "y_std": stats.y_std.flatten().tolist(),
        },
    }
    write_json(run_dir / "config.json", config)

    best_val_loss = float("inf")
    best_epoch = -1
    start_time = time.time()
    print(f"Run: {run_name}")
    print(f"Device: {device}")
    print(f"Parameters: {count_parameters(model):,}")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, loaders["train"], optimizer, criterion, device)
        val_metrics = evaluate(model, loaders["val"], criterion, device, stats_device)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_mse": val_metrics["mse"],
            "val_rel_l2": val_metrics["rel_l2"],
            "val_max_abs": val_metrics["max_abs"],
            "lr": optimizer.param_groups[0]["lr"],
            "elapsed_sec": time.time() - start_time,
        }
        append_history(run_dir / "history.csv", row)

        print(
            f"epoch {epoch:03d} | "
            f"train_loss {train_loss:.6f} | "
            f"val_loss {val_metrics['loss']:.6f} | "
            f"val_rel_l2 {val_metrics['rel_l2']:.4f} | "
            f"val_max_abs {val_metrics['max_abs']:.4f}"
        )

        latest_path = args.checkpoint_dir / f"{run_name}_latest.pt"
        torch.save(
            checkpoint_payload(model, optimizer, epoch, args, stats, val_metrics),
            latest_path,
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            best_path = args.checkpoint_dir / f"{run_name}_best.pt"
            torch.save(
                checkpoint_payload(model, optimizer, epoch, args, stats, val_metrics),
                best_path,
            )

    final_metrics = {"best_epoch": best_epoch, "best_val_loss": best_val_loss}
    best_checkpoint = torch.load(
        args.checkpoint_dir / f"{run_name}_best.pt",
        map_location=device,
    )
    model.load_state_dict(best_checkpoint["model_state_dict"])

    for split in ["val", "test", "ood_corner", "ood_multi"]:
        split_metrics = evaluate(model, loaders[split], criterion, device, stats_device)
        final_metrics[split] = split_metrics
        print(
            f"{split:>10s} | mse {split_metrics['mse']:.6f} | "
            f"rel_l2 {split_metrics['rel_l2']:.4f} | "
            f"max_abs {split_metrics['max_abs']:.4f}"
        )

    write_json(run_dir / "metrics.json", final_metrics)
    print(f"Saved logs to {run_dir}")
    print(f"Saved checkpoints to {args.checkpoint_dir}")


if __name__ == "__main__":
    main()
