import math
import random
import time
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning, module="spconv")
warnings.filterwarnings("ignore", category=FutureWarning, module="timm")

import numpy as np
import torch
from terraseg_dataloading import (
    DEFAULT_AUG_CFG,
    GROUP_PROBS,
    BalancedGroupSampler,
    build_group_indices,
    collate_scans,
)
from terraseg_loss import BCELovaszPerScan
from terraseg_utils import evaluate_loader, miou_vs_threshold
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.dataset import OmniLiDARDataset
from utils.splits import eval_splits, train_splits

from terraseg import build_terraseg, replace_bn1d_with_gn

# Paths.
results_root = Path("PUT_YOUR_DIRECTORY_HERE/TerraSeg_results")
omnilidar_root = Path("PUT_YOUR_DIRECTORY_HERE/OmniLiDAR")
pseudolabels_root = Path("PUT_YOUR_DIRECTORY_HERE/PseudoLabeler")

assert str(results_root) != "PUT_YOUR_DIRECTORY_HERE/TerraSeg_results", (
    "Folder for storing TerraSeg results. Change to directory in your file system!"
)
assert str(omnilidar_root) != "PUT_YOUR_DIRECTORY_HERE/OmniLiDAR", (
    "Directory to OmniLiDAR dataset. Change to directory in your file system!"
)
assert str(pseudolabels_root) != "PUT_YOUR_DIRECTORY_HERE/PseudoLabeler", (
    "Directory to pseudo-labels. Change to directory in your file system!"
)

results_root.mkdir(exist_ok=True, parents=True)
omnilidar_root.mkdir(exist_ok=True, parents=True)
pseudolabels_root.mkdir(exist_ok=True, parents=True)

# Released TerraSeg variant to train. Set to "B" for the accurate Base model (~46M params)
# or "S" for the efficient Small model (~12M params). All other hyperparameters are shared.
VARIANT = "B"

# Hyperparameters.
METHOD_NAME = f"TerraSeg-{VARIANT}"
SEED = 42
BATCH_SCANS = 2 if VARIANT == "B" else 8
NUM_WORKERS = 8
GRID_SIZE = 0.05
IGNORE_LABEL = 2
SAMPLES_PER_EPOCH = 51_200

EPOCHS = 10
LR = 1e-3
WEIGHT_DECAY = 5e-3
MAX_GRAD_NORM = 2.5
GRAD_ACCUM_STEPS = 256 if VARIANT == "B" else 64  # Effective batch = BATCH_SCANS * GRAD_ACCUM_STEPS = 512.
WARMUP_EPOCHS = 2
LOVASZ_WEIGHT = 1.0
INPUT_DIM = 3

DECISION_THRES_INIT = 0.50
DECISION_THRES_EMA_DECAY = 0.95
POS_WEIGHT_INIT = 1.00
POS_WEIGHT_EMA_DECAY = 0.90
GND_FRACTION_INIT = 0.50

EARLY_STOP_PATIENCE = 5

# Device and determinism.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

# Results directory.
timestamp = int(time.time())
run_dir = results_root / METHOD_NAME / str(timestamp)
run_dir.mkdir(exist_ok=True, parents=True)
log_path = run_dir / "log.txt"
tb_writer = SummaryWriter(log_dir=str(run_dir / "tb"))


def log_msg(msg: str) -> None:
    """
    Print a message and append it to the run log file.

    Args:
        msg (str) : Message to log.
    """
    print(msg)
    with open(log_path, "a") as f:
        f.write(msg + "\n")


log_msg(f"\nStarting new run at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} in {run_dir}\n")

# Datasets.
dataset_train = OmniLiDARDataset(
    root=omnilidar_root,
    splits=train_splits,
    pseudolabels_root=pseudolabels_root,
    remove_ego_points=False,
    compute_scans_metadata=False,
)
val_datasets = {
    dataset_name: OmniLiDARDataset(
        root=omnilidar_root,
        splits={dataset_name: split_name},
        pseudolabels_root="",
        remove_ego_points=False,
        compute_scans_metadata=False,
    )
    for dataset_name, split_name in eval_splits.items()
}

log_msg(repr(dataset_train) + "\n")
for dataset_name, val_dataset in val_datasets.items():
    log_msg(repr(val_dataset) + "\n")

# Balanced group sampler.
group_indices, group_probs_used = build_group_indices(
    dataset_pairs=dataset_train.pairs,
    cum_blocks=dataset_train.cum_blocks,
    group_probs=GROUP_PROBS,
)
train_sampler = BalancedGroupSampler(
    group_indices=group_indices,
    group_probs=group_probs_used,
    total_samples=SAMPLES_PER_EPOCH,
    rng_seed=SEED,
)

# Dataloaders.
train_loader = DataLoader(
    dataset=dataset_train,
    batch_size=BATCH_SCANS,
    sampler=train_sampler,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=True,
    collate_fn=lambda samples: collate_scans(
        samples=samples,
        grid_size=GRID_SIZE,
        ignore_label=IGNORE_LABEL,
        is_train=True,
        aug_cfg=DEFAULT_AUG_CFG,
    ),
    drop_last=False,
)
val_loaders = {
    dataset_name: DataLoader(
        dataset=val_dataset,
        batch_size=BATCH_SCANS,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=lambda samples: collate_scans(
            samples=samples,
            grid_size=GRID_SIZE,
            ignore_label=IGNORE_LABEL,
            is_train=False,
            aug_cfg=None,
        ),
        drop_last=False,
    )
    for dataset_name, val_dataset in val_datasets.items()
}

# Model.
model = build_terraseg(variant=VARIANT, input_dim=INPUT_DIM, num_classes=1)
model = replace_bn1d_with_gn(model=model, groups_default=32)
model = model.to(device)

# Optimizer.
param_groups = [
    {"params": model.backbone.parameters(), "lr": LR},
    {"params": model.cls_head.parameters(), "lr": LR},
]
optimizer = AdamW(param_groups, weight_decay=WEIGHT_DECAY)

# Scheduler: linear warmup (1 epoch) -> cosine decay, indexed by optimizer steps.
steps_per_epoch = max(math.ceil(len(train_loader) / GRAD_ACCUM_STEPS), 1)
warmup_iters = WARMUP_EPOCHS * steps_per_epoch
total_iters = EPOCHS * steps_per_epoch


def lr_lambda(step: int) -> float:
    """
    Compute the LR multiplier at a given optimizer step.

    Args:
        step (int) : Current optimizer step index (0-based).

    Returns:
        multiplier (float) : Multiplier applied to the base learning rate.
    """
    if step < warmup_iters:
        warmup_start = 0.01  # Start the warmup at 1% of the base LR.
        return warmup_start + (1.0 - warmup_start) * (step / max(warmup_iters, 1))
    progress = (step - warmup_iters) / max(total_iters - warmup_iters, 1)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


scheduler = LambdaLR(optimizer=optimizer, lr_lambda=lr_lambda)

# Loss.
loss_object = BCELovaszPerScan(
    lovasz_weight=LOVASZ_WEIGHT,
    ignore_label=IGNORE_LABEL,
    pos_weight_init=POS_WEIGHT_INIT,
).to(device)

# State.
global_step = 0  # Counts optimizer steps (i.e. after accumulation).
decision_thres = DECISION_THRES_INIT
ema_gnd_fraction = GND_FRACTION_INIT
ema_miou_vs_thres = torch.zeros((101,), dtype=torch.float32, device=device)
best_val_miou = 0.0
best_epoch = -1
patience_left = EARLY_STOP_PATIENCE

log_msg(
    f"Hyperparameters | epochs={EPOCHS} batch_scans={BATCH_SCANS} "
    f"grad_accum={GRAD_ACCUM_STEPS} effective_batch={BATCH_SCANS * GRAD_ACCUM_STEPS} "
    f"lr={LR:.1e} weight_decay={WEIGHT_DECAY:.1e} max_grad_norm={MAX_GRAD_NORM} "
    f"lovasz_weight={LOVASZ_WEIGHT}"
)
log_msg(
    f"Train sampler | samples_per_epoch={SAMPLES_PER_EPOCH} steps_per_epoch={steps_per_epoch} "
    f"total_iters={total_iters} warmup_iters={warmup_iters}\n"
)

# Training loop.
for epoch in range(1, EPOCHS + 1):
    epoch_loss_sum = 0.0
    num_optimizer_steps_epoch = 0

    accum_batch_loss = 0.0
    accum_correct = 0.0
    accum_ground = 0.0
    accum_num_points = 0
    accum_bce = 0.0
    accum_lovasz = 0.0

    pred_probs_list: list[torch.Tensor] = []
    target_labels_list: list[torch.Tensor] = []

    model.train()
    optimizer.zero_grad(set_to_none=True)

    for batch_idx, batch in enumerate(train_loader):
        # Move tensors to device.
        for key in ("coord", "feat", "offset", "target", "segment"):
            batch[key] = batch[key].to(device, non_blocking=True)

        bool_valid = batch["segment"] != IGNORE_LABEL
        target_labels = (batch["segment"][bool_valid] == 1).long()

        # Forward + loss.
        if batch["coord"].shape[0] == 0:
            log_msg(
                f"[{datetime.now().strftime('%H:%M:%S')}] "
                f"epoch {epoch:02d} batch_idx {batch_idx} | skipped empty batch after collate"
            )
            continue
        pred_logits = model(batch)
        loss_dict = loss_object(
            pred_logits=pred_logits, target_labels=batch["segment"], offset=batch["offset"]
        )
        loss = loss_dict["loss"] / GRAD_ACCUM_STEPS

        # Backward.
        loss.backward()

        # Accumulate statistics for logging.
        with torch.no_grad():
            pred_probs = torch.sigmoid(pred_logits[bool_valid])
            pred_labels = pred_probs > decision_thres

            loss_magnitude = float(loss.item()) * GRAD_ACCUM_STEPS
            num_points = int(bool_valid.sum().item())
            accuracy = (
                float((pred_labels == target_labels.bool()).float().mean().item())
                if target_labels.numel() > 0
                else 0.0
            )
            ground_fraction = (
                float((target_labels == 0).float().mean().item())
                if target_labels.numel() > 0
                else 0.0
            )

            accum_batch_loss += loss_magnitude
            accum_correct += accuracy * num_points
            accum_ground += ground_fraction * num_points
            accum_num_points += num_points
            accum_bce += float(loss_dict["bce"])
            accum_lovasz += float(loss_dict["lovasz"])

            pred_probs_list.append(pred_probs.detach())
            target_labels_list.append(target_labels.detach())

        is_last_minibatch = (batch_idx + 1) == len(train_loader)
        is_step_boundary = (batch_idx + 1) % GRAD_ACCUM_STEPS == 0
        if not (is_step_boundary or is_last_minibatch):
            continue

        # Optimizer step.
        if MAX_GRAD_NORM is not None and MAX_GRAD_NORM > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        global_step += 1

        # Decision threshold via EMA over mIoU-vs-threshold curves.
        pred_probs_all = torch.cat(pred_probs_list, dim=0)
        target_labels_all = torch.cat(target_labels_list, dim=0)
        thresholds, miou_curve, _, _ = miou_vs_threshold(
            pred_probs=pred_probs_all,
            target_labels=target_labels_all,
            ignore_label=IGNORE_LABEL,
        )
        ema_miou_vs_thres = (
            DECISION_THRES_EMA_DECAY * ema_miou_vs_thres
            + (1.0 - DECISION_THRES_EMA_DECAY) * miou_curve
        )
        best_idx = int(ema_miou_vs_thres.argmax().item())
        decision_thres = float(thresholds[best_idx].item())

        # Dynamic BCE positive-class weight via EMA over ground fraction.
        step_batch_loss = accum_batch_loss / GRAD_ACCUM_STEPS
        step_bce = accum_bce / GRAD_ACCUM_STEPS
        step_lovasz = accum_lovasz / GRAD_ACCUM_STEPS
        step_accuracy = accum_correct / max(accum_num_points, 1)
        step_ground_fraction = accum_ground / max(accum_num_points, 1)

        if math.isfinite(step_ground_fraction):
            ema_gnd_fraction = (
                POS_WEIGHT_EMA_DECAY * ema_gnd_fraction
                + (1.0 - POS_WEIGHT_EMA_DECAY) * step_ground_fraction
            )
        ema_gnd_fraction = min(max(ema_gnd_fraction, 0.0), 1.0 - 1e-6)
        dynamic_pos_weight = ema_gnd_fraction / max(1.0 - ema_gnd_fraction, 1e-6)
        loss_object.set_pos_weight(dynamic_pos_weight)

        # Logging.
        current_lr = optimizer.param_groups[0]["lr"]
        log_msg(
            f"[{datetime.now().strftime('%H:%M:%S')}] "
            f"epoch {epoch:02d} step {global_step:06d} | "
            f"loss {step_batch_loss:.4f} bce {step_bce:.4f} lovasz {step_lovasz:.4f} | "
            f"acc {step_accuracy:.3f} ema_gnd_fraction {ema_gnd_fraction:.3f} "
            f"pos_weight {dynamic_pos_weight:.3f} points {(accum_num_points / 1e6):.1f}M "
            f"lr {current_lr:.2e} dec_thres {decision_thres:.3f}"
        )

        tb_writer.add_scalar("train/loss", step_batch_loss, global_step)
        tb_writer.add_scalar("train/bce", step_bce, global_step)
        tb_writer.add_scalar("train/lovasz", step_lovasz, global_step)
        tb_writer.add_scalar("train/accuracy", step_accuracy, global_step)
        tb_writer.add_scalar("train/lr", current_lr, global_step)
        tb_writer.add_scalar("train/pos_weight", dynamic_pos_weight, global_step)
        tb_writer.add_scalar("train/ema_gnd_fraction", ema_gnd_fraction, global_step)
        tb_writer.add_scalar("train/decision_thres", decision_thres, global_step)
        tb_writer.add_scalar("train/points_per_step", accum_num_points, global_step)

        epoch_loss_sum += step_batch_loss
        num_optimizer_steps_epoch += 1

        # Reset accumulators.
        accum_batch_loss = 0.0
        accum_correct = 0.0
        accum_ground = 0.0
        accum_num_points = 0
        accum_bce = 0.0
        accum_lovasz = 0.0
        pred_probs_list = []
        target_labels_list = []

    avg_train_loss = epoch_loss_sum / max(num_optimizer_steps_epoch, 1)
    log_msg(
        f"[{datetime.now().strftime('%H:%M:%S')}] "
        f"EPOCH {epoch:02d} train | loss {avg_train_loss:.4f}"
    )
    tb_writer.add_scalar("epoch/train_loss", avg_train_loss, epoch)

    # Validation on the model, across all eval splits.
    mious: list[float] = []
    for dataset_name, val_loader in val_loaders.items():
        metrics = evaluate_loader(
            model=model,
            val_loader=val_loader,
            device=device,
            decision_thres=decision_thres,
            ignore_label=IGNORE_LABEL,
        )

        mious.append(metrics["miou"])

        log_msg(
            f"[{datetime.now().strftime('%H:%M:%S')}] EPOCH {epoch:02d} validation {dataset_name} "
            f"| recall0 {100 * metrics['recall0']:.2f} "
            f"precision0 {100 * metrics['precision0']:.2f} "
            f"IoU0 {100 * metrics['iou0']:.2f} "
            f"recall1 {100 * metrics['recall1']:.2f} "
            f"precision1 {100 * metrics['precision1']:.2f} "
            f"IoU1 {100 * metrics['iou1']:.2f} "
            f"mIoU {100 * metrics['miou']:.2f} "
        )

        prefix = "val"
        tb_writer.add_scalar(f"{prefix}/{dataset_name}/recall0", metrics["recall0"], epoch)
        tb_writer.add_scalar(f"{prefix}/{dataset_name}/precision0", metrics["precision0"], epoch)
        tb_writer.add_scalar(f"{prefix}/{dataset_name}/iou0", metrics["iou0"], epoch)
        tb_writer.add_scalar(f"{prefix}/{dataset_name}/recall1", metrics["recall1"], epoch)
        tb_writer.add_scalar(f"{prefix}/{dataset_name}/precision1", metrics["precision1"], epoch)
        tb_writer.add_scalar(f"{prefix}/{dataset_name}/iou1", metrics["iou1"], epoch)
        tb_writer.add_scalar(f"{prefix}/{dataset_name}/miou", metrics["miou"], epoch)

    mean_miou = float(np.mean(mious))
    tb_writer.add_scalar("val/mean_miou", mean_miou, epoch)

    log_msg(
        f"[{datetime.now().strftime('%H:%M:%S')}] EPOCH {epoch:02d} mean validation mIoU "
        f"| {100 * mean_miou:.2f}"
    )

    # Early stopping & checkpointing on the model's mean validation mIoU.
    if mean_miou > best_val_miou:
        best_val_miou = mean_miou
        best_epoch = epoch
        patience_left = EARLY_STOP_PATIENCE
        torch.save(
            {
                "epoch": epoch,
                "global_step": global_step,
                "model_state_dict": model.state_dict(),
                "decision_thres": decision_thres,
                "mean_val_miou": mean_miou,
            },
            run_dir / "best.pth",
        )
        log_msg(
            f"[{datetime.now().strftime('%H:%M:%S')}] "
            f"Saved new best checkpoint (mean mIoU = {100 * best_val_miou:.2f})"
        )
    else:
        patience_left -= 1
        log_msg(
            f"[{datetime.now().strftime('%H:%M:%S')}] "
            f"No improvement on mean mIoU ({100 * mean_miou:.2f} <= "
            f"{100 * best_val_miou:.2f}); patience_left = {patience_left}"
        )
        if patience_left <= 0:
            log_msg(
                f"[{datetime.now().strftime('%H:%M:%S')}] "
                f"Early stopping triggered at epoch {epoch:02d}."
            )
            break

# Summary.
log_msg(
    f"\nTraining completed. Best mean validation mIoU: {100 * best_val_miou:.2f} "
    f"at epoch {best_epoch:02d}."
)
tb_writer.close()
