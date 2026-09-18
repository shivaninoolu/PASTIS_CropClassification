import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import (
    BATCH_SIZE, EPOCHS, LEARNING_RATE, NUM_WORKERS,
    EARLY_STOPPING, PATIENCE, MIN_DELTA,
    USE_SCHEDULER, SCHEDULER_FACTOR, SCHEDULER_PATIENCE,
    SCHEDULER_MIN_LR, TEMPORAL_FEATURES, SEED,
    CHECKPOINT_DIR, SCENARIOS,
)
from dataset import (
    load_metadata, build_sample_pairs, get_fold_indices,
    compute_statistics, PASTISDataset, NUM_CLASSES, IGNORE_INDEX,
)
from feature_selection import select_features
from model import build_model


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def metrics_from_logits(logits, target):
    pred = logits.argmax(dim=1)
    valid = target != IGNORE_INDEX

    if valid.sum() == 0:
        return 0.0, 0.0

    p = pred[valid]
    y = target[valid]
    accuracy = (p == y).float().mean().item()

    ious = []
    for c in range(NUM_CLASSES):
        pc = p == c
        yc = y == c
        union = (pc | yc).sum().item()
        if union > 0:
            inter = (pc & yc).sum().item()
            ious.append(inter / union)

    miou = float(np.mean(ious)) if ious else 0.0
    return accuracy, miou


def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train(train)
    total_loss = 0.0
    accs, mious = [], []

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.set_grad_enabled(train):
            logits = model(images)
            loss = criterion(logits, targets)

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

        total_loss += loss.item() * images.size(0)
        acc, miou = metrics_from_logits(logits.detach(), targets)
        accs.append(acc)
        mious.append(miou)

    n = len(loader.dataset)
    return total_loss / max(n, 1), float(np.mean(accs)), float(np.mean(mious))


def load_fold_class_weights(fold, device):
    path = Path(CHECKPOINT_DIR).parent / "class_weights" / f"class_weights_fold_{fold}.npy"
    if not path.exists():
        raise FileNotFoundError(
            f"Class weights not found: {path}\n"
            "Run your existing class_weights.py first."
        )

    w = np.load(path).astype(np.float32)

    if len(w) != NUM_CLASSES:
        raise ValueError(
            f"Expected {NUM_CLASSES} class weights, got {len(w)}."
        )

    # Absent classes have weight 0. CrossEntropyLoss accepts this, but
    # they cannot be learned because they have no training pixels.
    return torch.tensor(w, dtype=torch.float32, device=device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--scenario", choices=SCENARIOS.keys(), default="baseline")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    args = parser.parse_args()

    seed_everything()

    epochs = EPOCHS if args.epochs is None else args.epochs
    batch_size = BATCH_SIZE if args.batch_size is None else args.batch_size
    lr = LEARNING_RATE if args.lr is None else args.lr

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    metadata = load_metadata()
    pairs = build_sample_pairs(metadata=metadata)
    train_idx, val_idx = get_fold_indices(pairs, args.fold)

    print(f"Scenario: {args.scenario}")
    print(f"Fold: {args.fold}")
    print(f"Training samples: {len(train_idx)}")
    print(f"Validation samples: {len(val_idx)}")

    selected_features = None
    if SCENARIOS[args.scenario]["feature_selection"]:
        selected_features = select_features(
            pairs, train_idx, args.scenario, args.fold
        )

    mean, std = compute_statistics(
        pairs,
        train_idx,
        scenario=args.scenario,
        selected_features=selected_features,
    )

    train_ds = PASTISDataset(
        pairs, train_idx, mean, std,
        scenario=args.scenario,
        feature_names=selected_features,
    )
    val_ds = PASTISDataset(
        pairs, val_idx, mean, std,
        scenario=args.scenario,
        feature_names=selected_features,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    n_bands = len(selected_features) if selected_features else (
        14 if SCENARIOS[args.scenario]["use_indices"] else 10
    )

    model = build_model(
        SCENARIOS[args.scenario]["model"],
        n_bands=n_bands,
        n_classes=NUM_CLASSES,
        temporal_features=TEMPORAL_FEATURES,
    ).to(device)

    print(f"Input channels: {n_bands}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    class_weights = load_fold_class_weights(args.fold, device)

    train_criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        ignore_index=IGNORE_INDEX,
    )
    
    val_criterion = nn.CrossEntropyLoss(
        ignore_index=IGNORE_INDEX,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    scheduler = None
    if USE_SCHEDULER:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=SCHEDULER_FACTOR,
            patience=SCHEDULER_PATIENCE,
            min_lr=SCHEDULER_MIN_LR,
        )

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CHECKPOINT_DIR / f"best_{args.scenario}_fold_{args.fold}.pth"

    best_miou = -1.0
    best_epoch = -1
    bad_epochs = 0

    for epoch in range(1, epochs + 1):
        train_loss, train_acc, train_miou = run_epoch(
            model,
            train_loader,
            train_criterion,
            optimizer,
            device,
            train=True
        )
        
        val_loss, val_acc, val_miou = run_epoch(
            model,
            val_loader,
            val_criterion,
            optimizer=None,
            device=device,
            train=False
        )

        if scheduler is not None:
            scheduler.step(val_miou)

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"train loss {train_loss:.4f} | "
            f"val loss {val_loss:.4f} | "
            f"val acc {val_acc:.4f} | "
            f"val mIoU {val_miou:.4f} | "
            f"lr {current_lr:.2e}"
        )

        if val_miou > best_miou + MIN_DELTA:
            best_miou = val_miou
            best_epoch = epoch
            bad_epochs = 0

            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_miou": val_miou,
                "val_accuracy": val_acc,
                "mean": mean,
                "std": std,
                "validation_fold": args.fold,
                "scenario": args.scenario,
                "selected_features": selected_features,
                "n_bands": n_bands,
                "num_classes": NUM_CLASSES,
                "ignore_index": IGNORE_INDEX,
                "class_weights": class_weights.cpu(),
            }, ckpt_path)

            print(f"  Saved best checkpoint: {ckpt_path}")
        else:
            bad_epochs += 1

        if EARLY_STOPPING and bad_epochs >= PATIENCE:
            print(
                f"Early stopping at epoch {epoch}. "
                f"Best epoch={best_epoch}, best mIoU={best_miou:.4f}"
            )
            break

    print("\nTraining complete.")
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation mIoU: {best_miou:.4f}")
    print(f"Checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
