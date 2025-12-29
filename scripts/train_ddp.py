"""Distributed training script for voxel autoregressive model."""
import os
import json
import time
import argparse
import signal

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from contextlib import nullcontext


class TimeoutError(Exception):
    """Exception raised when a function times out."""
    pass


def timeout_handler(signum, frame):
    """Signal handler for timeout."""
    raise TimeoutError("Function timed out")

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from functools import partial
from voxel_ode.dataset import VoxelARIndexDataset, RawH5Dataset, pad_collate_fn
from voxel_ode.model import VoxelAutoRegressor, LinearBaseline
from voxel_ode.schedulers import WarmupCosine
from voxel_ode.utils import set_seed, ddp_print
from voxel_ode.file_aware_sampler import DDPFileGroupedBatchSampler


def init_distributed():
    """Initialize DDP if running with torchrun."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        import datetime
        # Disable P2P for Docker environments where it may not work
        os.environ['NCCL_P2P_DISABLE'] = '1'
        os.environ['NCCL_IB_DISABLE'] = '1'
        dist.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=1800))
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        return True
    return False


def load_stats(prepped_root):
    """Load statistics JSON."""
    with open(os.path.join(prepped_root, "stats.json"), "r") as f:
        stats = json.load(f)
    return stats


def evaluate_rollout(model, raw_h5_dir, stats, device, fixed_pad_size=None, test_files=None, max_steps=29,
                     acc_abs_p=5.0, acc_abs_t=5.0, acc_abs_wept=1e10):
    """
    Perform full autoregressive rollout evaluation on test files (NO teacher forcing).

    This is the TRUE test of model quality - we feed model predictions back as inputs
    to measure drift over time. This is what we really care about for deployment.

    NOTE: This function processes data at native resolution (no padding) because:
    1. The model is fully convolutional and can handle variable input sizes
    2. All test files have the same dimensions
    3. This matches rollout_inference.py behavior

    Args:
        model: The model (DDP-wrapped or not)
        raw_h5_dir: Directory with raw h5 files
        stats: Statistics dict for normalization
        device: CUDA device
        fixed_pad_size: Ignored - kept for API compatibility
        test_files: List of test file names (default: v2.5_0001.h5 to v2.5_0005.h5)
        max_steps: Max rollout steps (default 29 for 30-step prediction)

    Returns:
        Dictionary with rollout MSE and Acc5 metrics averaged across files and timesteps
    """
    import h5py
    import numpy as np

    model.eval()

    if test_files is None:
        test_files = [f"v2.5_{i:04d}.h5" for i in range(1, 6)]  # v2.5_0001.h5 to v2.5_0005.h5

    # Load stats for normalization
    static_mean = torch.tensor(stats["mean"]["static"], device=device).view(-1, 1, 1, 1)
    static_std = torch.tensor(stats["std"]["static"], device=device).view(-1, 1, 1, 1).clamp(min=1e-6)
    grid_input_mean = torch.tensor(stats["mean"]["grid_input_absolute"], device=device)
    grid_input_std = torch.tensor(stats["std"]["grid_input_absolute"], device=device).clamp(min=1e-6)
    grid_output_std = torch.tensor(stats["std"]["grid_output_delta"], device=device).clamp(min=1e-6)

    STATIC_KEYS = ["FaultId", "InjRate", "IsActive", "IsWell", "PermX", "PermY", "PermZ", "Porosity"]

    # Accumulators for metrics across all files
    all_mse_p, all_mse_t, all_mse_w = [], [], []
    all_acc5_p, all_acc5_t, all_acc5_w = [], [], []
    files_processed = 0

    # Get underlying model (handle DDP wrapping)
    base_model = model.module if hasattr(model, 'module') else model

    with torch.no_grad():
        for fname in test_files:
            fpath = os.path.join(raw_h5_dir, fname)
            if not os.path.exists(fpath):
                continue

            # Load simulation data
            with h5py.File(fpath, "r") as f:
                static = np.stack([f[f"Input/{k}"][...] for k in STATIC_KEYS], axis=0)
                params = f["Input/ParamsScalar"][:]
                p_true = f["Output/Pressure"][:]
                t_true = f["Output/Temperature"][:]
                w_true = f["Output/WEPT"][:]

            T, Z, Y, X = p_true.shape

            # Find valid timesteps
            max_valid_t = 0
            for t in range(T):
                if (p_true[t] != -999).any():
                    max_valid_t = t

            rollout_steps = min(max_steps, max_valid_t)
            if rollout_steps < 2:
                continue

            # Normalize static - process at native resolution (no padding)
            static_t = torch.from_numpy(static).float().to(device)
            static_norm = (static_t - static_mean) / static_std
            static_norm[static_t == -999] = 0
            static_norm = static_norm.unsqueeze(0)  # [1, 8, Z, Y, X]

            params_t = torch.from_numpy(params).float().to(device).unsqueeze(0)  # [1, 26]

            # Initialize AR state with ground truth t=0
            p_ar = torch.from_numpy(p_true[0]).float().to(device)
            t_ar = torch.from_numpy(t_true[0]).float().to(device)
            w_ar = torch.from_numpy(w_true[0]).float().to(device)

            mask_p = (p_ar != -999)
            mask_t = (t_ar != -999)
            mask_w = (w_ar != -999)

            # Per-file accumulators
            file_mse_p, file_mse_t, file_mse_w = [], [], []
            file_acc5_p, file_acc5_t, file_acc5_w = [], [], []

            # Full AR rollout
            prev_p, prev_t, prev_w = torch.zeros_like(p_ar), torch.zeros_like(t_ar), torch.zeros_like(w_ar)
            curr_p, curr_t, curr_w = p_ar.clone(), t_ar.clone(), w_ar.clone()

            for step in range(rollout_steps):
                # Build input [static(8), P_tm1, T_tm1, W_tm1, P_t, T_t, W_t]
                x_input = torch.zeros(1, 14, Z, Y, X, device=device, dtype=torch.float32)
                x_input[0, :8] = static_norm[0]

                # Normalize inputs
                x_input[0, 8] = (prev_p - grid_input_mean[0]) / grid_input_std[0]
                x_input[0, 9] = (prev_t - grid_input_mean[1]) / grid_input_std[1]
                x_input[0, 10] = (prev_w - grid_input_mean[2]) / grid_input_std[2]
                x_input[0, 11] = (curr_p - grid_input_mean[0]) / grid_input_std[0]
                x_input[0, 12] = (curr_t - grid_input_mean[1]) / grid_input_std[1]
                x_input[0, 13] = (curr_w - grid_input_mean[2]) / grid_input_std[2]

                # Mask sentinels
                x_input[0, 8][~mask_p] = 0
                x_input[0, 9][~mask_t] = 0
                x_input[0, 10][~mask_w] = 0
                x_input[0, 11][~mask_p] = 0
                x_input[0, 12][~mask_t] = 0
                x_input[0, 13][~mask_w] = 0

                # Forward pass (no padding - native resolution)
                with torch.cuda.amp.autocast(enabled=True):
                    grid_pred_norm, _ = base_model(x_input, params_t)

                # FIX: Guard against NaN/inf in predictions (prevent rollout explosion)
                grid_pred_norm = torch.nan_to_num(grid_pred_norm, nan=0.0, posinf=10.0, neginf=-10.0)
                grid_pred_norm = grid_pred_norm.clamp(-10.0, 10.0)  # Clamp normalized residuals

                # Convert to raw deltas
                delta_p = grid_pred_norm[0, 0] * grid_output_std[0]
                delta_t = grid_pred_norm[0, 1] * grid_output_std[1]
                delta_w = grid_pred_norm[0, 2] * grid_output_std[2]

                # Update AR state: current -> previous, prediction -> current
                prev_p, prev_t, prev_w = curr_p.clone(), curr_t.clone(), curr_w.clone()
                curr_p = torch.where(mask_p, curr_p + delta_p, curr_p)
                curr_t = torch.where(mask_t, curr_t + delta_t, curr_t)
                curr_w = torch.where(mask_w, curr_w + delta_w, curr_w)

                # FIX: Clamp absolute state to physically reasonable ranges
                # Pressure: 0 to 1000 bar (typical geothermal range)
                # Temperature: -50 to 500 C
                # WEPT: 0 to 1e15 J (cumulative energy)
                curr_p = torch.nan_to_num(curr_p, nan=0.0, posinf=1000.0, neginf=0.0)
                curr_t = torch.nan_to_num(curr_t, nan=0.0, posinf=500.0, neginf=-50.0)
                curr_w = torch.nan_to_num(curr_w, nan=0.0, posinf=1e15, neginf=0.0)
                curr_p = curr_p.clamp(0.0, 1000.0)
                curr_t = curr_t.clamp(-50.0, 500.0)
                # FIX: Lowered WEPT clamp from 1e15 to 1e13 (typical WEPT ~1e12)
                curr_w = curr_w.clamp(0.0, 1e13)

                # Compute metrics at this timestep vs ground truth
                target_t = step + 1
                if target_t <= max_valid_t:
                    gt_p = torch.from_numpy(p_true[target_t]).float().to(device)
                    gt_t = torch.from_numpy(t_true[target_t]).float().to(device)
                    gt_w = torch.from_numpy(w_true[target_t]).float().to(device)

                    # FIX: Safe MSE computation handling empty masks
                    def safe_mse(pred, true, mask):
                        if mask.sum() == 0:
                            return None
                        diff = pred[mask] - true[mask]
                        return (diff * diff).mean().item()

                    mse_p = safe_mse(curr_p, gt_p, mask_p)
                    mse_t = safe_mse(curr_t, gt_t, mask_t)
                    mse_w = safe_mse(curr_w, gt_w, mask_w)

                    # Skip this timestep if any MSE is None (empty mask)
                    if mse_p is None or mse_t is None or mse_w is None:
                        continue

                    # Acc5 using ABSOLUTE error thresholds (not relative)
                    def acc5_abs(pred, true, mask, threshold):
                        pm, tm = pred[mask], true[mask]
                        if len(pm) == 0:
                            return 0.0
                        abs_err = (pm - tm).abs()
                        return (abs_err <= threshold).float().mean().item()

                    # No need to exclude near-zero cells with absolute thresholds
                    acc5_p = acc5_abs(curr_p, gt_p, mask_p, acc_abs_p)
                    acc5_t = acc5_abs(curr_t, gt_t, mask_t, acc_abs_t)
                    acc5_w = acc5_abs(curr_w, gt_w, mask_w, acc_abs_wept)

                    file_mse_p.append(mse_p)
                    file_mse_t.append(mse_t)
                    file_mse_w.append(mse_w)
                    file_acc5_p.append(acc5_p)
                    file_acc5_t.append(acc5_t)
                    file_acc5_w.append(acc5_w)

            # Average across timesteps for this file
            if len(file_mse_p) > 0:
                all_mse_p.append(np.mean(file_mse_p))
                all_mse_t.append(np.mean(file_mse_t))
                all_mse_w.append(np.mean(file_mse_w))
                all_acc5_p.append(np.mean(file_acc5_p))
                all_acc5_t.append(np.mean(file_acc5_t))
                all_acc5_w.append(np.mean(file_acc5_w))
                files_processed += 1

    # Average across all files
    if files_processed == 0:
        return {"rollout/mse_p": 0.0, "rollout/mse_t": 0.0, "rollout/mse_w": 0.0,
                "rollout/acc5_p": 0.0, "rollout/acc5_t": 0.0, "rollout/acc5_w": 0.0}

    metrics = {
        "rollout/mse_p": np.mean(all_mse_p),
        "rollout/mse_t": np.mean(all_mse_t),
        "rollout/mse_w": np.mean(all_mse_w),
        "rollout/acc5_p": np.mean(all_acc5_p),
        "rollout/acc5_t": np.mean(all_acc5_t),
        "rollout/acc5_w": np.mean(all_acc5_w),
        "rollout/files_evaluated": files_processed,
    }

    return metrics


def evaluate(model, loader, device, stats, scalar_log1p_flags, max_batches=None, predict_grid_only=False, grid_out_channels=2,
             acc_abs_p=5.0, acc_abs_t=5.0, acc_abs_wept=1e10):
    """
    Evaluate model on test set.

    Computes:
    - MSE in whitened (normalized) space for loss tracking
    - Acc on ABSOLUTE values using absolute error thresholds (not relative %)

    For residual prediction:
    - Input contains normalized P_t, T_t, WEPT_t at specific channel indices
    - Output is normalized delta (residual)
    - To compute absolute prediction: abs_pred = abs_current + delta_pred
    - Acc checks: |abs_pred - abs_true| <= threshold (e.g., 5 bar for P, 5 C for T)

    Returns:
        Dictionary with per-output MSE and absolute accuracy metrics
    """
    model.eval()

    # Accumulators
    mse_accum = {
        "grid_p": 0.0,
        "grid_T": 0.0,
    }
    acc_accum = {
        "grid_p": 0.0,
        "grid_T": 0.0,
    }

    # Add WEPT accumulator if in v2.5 mode
    if grid_out_channels == 3:
        mse_accum["grid_wept"] = 0.0
        acc_accum["grid_wept"] = 0.0

    # Only track scalar metrics if in v2.4 and not in grid-only mode
    if not predict_grid_only and grid_out_channels == 2:
        mse_accum["scalar"] = torch.zeros(5, device=device)
        acc_accum["scalar"] = torch.zeros(5, device=device)

    # Load stats for unwhitening
    # For v2.5 raw_h5_mode: use grid_input_absolute for inputs, grid_output_delta for outputs
    # Fallback to "grid" key for backward compatibility
    num_grid_ch = len(stats["grid_channels"])

    # Stats for unnormalizing INPUT (current state P_t, T_t, WEPT_t)
    if "grid_input_absolute" in stats["mean"]:
        input_mean = torch.tensor(stats["mean"]["grid_input_absolute"], device=device, dtype=torch.float32)
        input_std = torch.tensor(stats["std"]["grid_input_absolute"], device=device, dtype=torch.float32).clamp(min=1e-6)
    else:
        # Fallback: use "grid" stats (these are absolute stats in current setup)
        input_mean = torch.tensor(stats["mean"]["grid"], device=device, dtype=torch.float32)
        input_std = torch.tensor(stats["std"]["grid"], device=device, dtype=torch.float32).clamp(min=1e-6)

    # Stats for unnormalizing OUTPUT (deltas)
    if "grid_output_delta" in stats["mean"]:
        delta_mean = torch.tensor(stats["mean"]["grid_output_delta"], device=device, dtype=torch.float32)
        delta_std = torch.tensor(stats["std"]["grid_output_delta"], device=device, dtype=torch.float32).clamp(min=1e-6)
    else:
        # Fallback: use "grid" stats
        delta_mean = torch.tensor(stats["mean"]["grid"], device=device, dtype=torch.float32)
        delta_std = torch.tensor(stats["std"]["grid"], device=device, dtype=torch.float32).clamp(min=1e-6)

    # Channel indices for extracting current state from input
    # Input structure: [static(8), P_{t-1}(1), T_{t-1}(1), WEPT_{t-1}(1), P_t(1), T_t(1), WEPT_t(1)]
    # So P_t is at index 11, T_t at 12, WEPT_t at 13
    C_static = len(stats["static_channels"])  # 8
    # Indices: static(0-7), P_t-1(8), T_t-1(9), WEPT_t-1(10), P_t(11), T_t(12), WEPT_t(13)
    idx_P_t = C_static + 3  # 11
    idx_T_t = C_static + 4  # 12
    idx_WEPT_t = C_static + 5  # 13

    if not predict_grid_only and grid_out_channels == 2:
        s_mean = torch.tensor(stats["mean"]["scalar"], device=device, dtype=torch.float32).view(1, 5)
        s_std = torch.tensor(stats["std"]["scalar"], device=device, dtype=torch.float32).view(1, 5).clamp(min=1e-6)

    count = 0

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if (max_batches is not None) and (i >= max_batches):
                break

            x_grid = batch["x_grid"].to(device, non_blocking=True)
            params = batch["params"].to(device, non_blocking=True)
            y_grid = batch["y_grid"].to(device, non_blocking=True)
            valid_mask = batch["valid_mask"].to(device, non_blocking=True)  # [B, Z, Y, X]

            grid_pred, scalar_pred = model(x_grid, params)

            # Compute masked MSE in WHITENED space (only on valid cells)
            se_p = (grid_pred[:, 0] - y_grid[:, 0]) ** 2
            se_T = (grid_pred[:, 1] - y_grid[:, 1]) ** 2

            # Apply mask and normalize by number of valid cells
            mse_p = ((se_p * valid_mask).sum() / valid_mask.sum().clamp(min=1)).detach()
            mse_T = ((se_T * valid_mask).sum() / valid_mask.sum().clamp(min=1)).detach()

            # ===== COMPUTE ACC5 ON ABSOLUTE VALUES =====
            # Extract current state (P_t, T_t, WEPT_t) from input and unnormalize
            # x_grid shape: [B, C_in, Z, Y, X]
            P_t_norm = x_grid[:, idx_P_t]  # [B, Z, Y, X]
            T_t_norm = x_grid[:, idx_T_t]  # [B, Z, Y, X]

            # Unnormalize current state to absolute values
            P_t_abs = P_t_norm * input_std[0] + input_mean[0]
            T_t_abs = T_t_norm * input_std[1] + input_mean[1]

            # Unnormalize deltas (predictions and targets)
            # Note: deltas are normalized as x / std (no mean subtraction) in dataset
            delta_P_pred = grid_pred[:, 0] * delta_std[0]
            delta_T_pred = grid_pred[:, 1] * delta_std[1]
            delta_P_true = y_grid[:, 0] * delta_std[0]
            delta_T_true = y_grid[:, 1] * delta_std[1]

            # Compute absolute predicted and true values
            P_pred_abs = P_t_abs + delta_P_pred
            T_pred_abs = T_t_abs + delta_T_pred
            P_true_abs = P_t_abs + delta_P_true
            T_true_abs = T_t_abs + delta_T_true

            def acc_ok_abs(pred, true, mask, threshold):
                """Compute masked absolute accuracy (|pred - true| <= threshold)."""
                correct = ((pred - true).abs() <= threshold).float()
                return (correct * mask).sum() / mask.sum().clamp(min=1)

            # No need to exclude near-zero cells with absolute thresholds
            acc_p = acc_ok_abs(P_pred_abs, P_true_abs, valid_mask, acc_abs_p)
            acc_T = acc_ok_abs(T_pred_abs, T_true_abs, valid_mask, acc_abs_t)

            # Accumulate grid metrics
            mse_accum["grid_p"] += mse_p.item()
            mse_accum["grid_T"] += mse_T.item()
            acc_accum["grid_p"] += acc_p.item()
            acc_accum["grid_T"] += acc_T.item()

            # Compute WEPT metrics if in v2.5 mode
            if grid_out_channels == 3:
                se_wept = (grid_pred[:, 2] - y_grid[:, 2]) ** 2
                mse_wept = ((se_wept * valid_mask).sum() / valid_mask.sum().clamp(min=1)).detach()

                # WEPT Acc on absolute values (using absolute threshold)
                WEPT_t_norm = x_grid[:, idx_WEPT_t]
                WEPT_t_abs = WEPT_t_norm * input_std[2] + input_mean[2]
                delta_WEPT_pred = grid_pred[:, 2] * delta_std[2]
                delta_WEPT_true = y_grid[:, 2] * delta_std[2]
                WEPT_pred_abs = WEPT_t_abs + delta_WEPT_pred
                WEPT_true_abs = WEPT_t_abs + delta_WEPT_true
                acc_wept = acc_ok_abs(WEPT_pred_abs, WEPT_true_abs, valid_mask, acc_abs_wept)

                mse_accum["grid_wept"] += mse_wept.item()
                acc_accum["grid_wept"] += acc_wept.item()

            # Only compute scalar metrics if in v2.4 and not in grid-only mode
            if not predict_grid_only and grid_out_channels == 2:
                y_scalar = batch["y_scalar"].to(device, non_blocking=True)

                # Scalars
                s_hat = scalar_pred * s_std + s_mean
                s_true = y_scalar * s_std + s_mean

                # Inverse log1p if needed
                for k, flag in enumerate(scalar_log1p_flags):
                    if flag:
                        s_hat[:, k] = torch.expm1(s_hat[:, k])
                        s_true[:, k] = torch.expm1(s_true[:, k])

                mse_s = torch.mean((scalar_pred - y_scalar) ** 2, dim=0).detach()
                acc_s = torch.mean(
                    ((s_hat - s_true).abs() <= 0.05 * s_true.abs().clamp(min=1e-8)).float(),
                    dim=0
                ).detach()

                # Accumulate scalar metrics
                mse_accum["scalar"] += mse_s
                acc_accum["scalar"] += acc_s

            count += 1

    # Reduce across GPUs if using DDP
    if dist.is_initialized():
        # Reduce grid metrics (always P and T)
        for k in ["grid_p", "grid_T"]:
            t = torch.tensor([mse_accum[k], acc_accum[k], count], device=device)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            mse_accum[k] = (t[0] / t[2]).item()
            acc_accum[k] = (t[1] / t[2]).item()

        # Reduce WEPT metrics if in v2.5 mode
        if grid_out_channels == 3:
            t = torch.tensor([mse_accum["grid_wept"], acc_accum["grid_wept"], count], device=device)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            mse_accum["grid_wept"] = (t[0] / t[2]).item()
            acc_accum["grid_wept"] = (t[1] / t[2]).item()

        if not predict_grid_only and grid_out_channels == 2:
            t_mse = torch.cat([mse_accum["scalar"], torch.tensor([count], device=device)])
            t_acc = torch.cat([acc_accum["scalar"], torch.tensor([count], device=device)])
            dist.all_reduce(t_mse, op=dist.ReduceOp.SUM)
            dist.all_reduce(t_acc, op=dist.ReduceOp.SUM)
            count = int(t_mse[-1].item())
            mse_accum["scalar"] = (t_mse[:-1] / max(1, count)).cpu().tolist()
            acc_accum["scalar"] = (t_acc[:-1] / max(1, count)).cpu().tolist()
    else:
        if not predict_grid_only and grid_out_channels == 2:
            mse_accum["scalar"] = (mse_accum["scalar"] / max(1, count)).cpu().tolist()
            acc_accum["scalar"] = (acc_accum["scalar"] / max(1, count)).cpu().tolist()
        for k in ["grid_p", "grid_T"]:
            mse_accum[k] = mse_accum[k] / max(1, count)
            acc_accum[k] = acc_accum[k] / max(1, count)
        if grid_out_channels == 3:
            mse_accum["grid_wept"] = mse_accum["grid_wept"] / max(1, count)
            acc_accum["grid_wept"] = acc_accum["grid_wept"] / max(1, count)

    # Format results
    metrics = {
        "mse/grid_pressure": mse_accum["grid_p"],
        "mse/grid_temperature": mse_accum["grid_T"],
        "acc5/grid_pressure": acc_accum["grid_p"],
        "acc5/grid_temperature": acc_accum["grid_T"],
    }

    # Add WEPT metrics if in v2.5 mode
    if grid_out_channels == 3:
        metrics["mse/grid_wept"] = mse_accum["grid_wept"]
        metrics["acc5/grid_wept"] = acc_accum["grid_wept"]

    # Only add scalar metrics if in v2.4 and not in grid-only mode
    if not predict_grid_only and grid_out_channels == 2:
        for i, name in enumerate(stats["scalar_channels"]):
            metrics[f"mse/{name}"] = mse_accum["scalar"][i]
            metrics[f"acc5/{name}"] = acc_accum["scalar"][i]

    return metrics


def scheduled_sampling_step(model, train_ds, device, sched_prob, stats, use_amp=True, fixed_pad_size=None, scaler=None, max_rollout_steps=5, global_step=0):
    """
    Perform one scheduled sampling step: multi-step rollout with mixed inputs.

    With probability `sched_prob`, use model's predicted state as input for next step.
    With probability `1-sched_prob`, use ground truth state.
    Target is ALWAYS ground truth delta.

    To avoid OOM, we only do `max_rollout_steps` forward+backward passes per call.

    IMPORTANT for DDP: All random decisions are seeded with global_step so all ranks
    make the same decisions. This ensures all GPUs execute the same code paths.

    Args:
        model: The model (in training mode) - should be DDP-wrapped for gradient sync
        train_ds: RawH5Dataset instance with get_trajectory() method
        device: CUDA device
        sched_prob: Probability of using model predictions (0.0 to 1.0)
        stats: Statistics dict for normalization
        use_amp: Whether to use automatic mixed precision
        fixed_pad_size: Optional (Z, Y, X) tuple for padding
        scaler: GradScaler for mixed precision
        max_rollout_steps: Max steps to unroll (to limit memory usage)
        global_step: Current training step (used to seed random for DDP sync)

    Returns:
        (total_loss, num_steps, loss_p, loss_T, loss_wept) - all are scalar tensors (detached)
    """
    import random
    import torch.distributed as dist

    # IMPORTANT for DDP: All ranks must select the same file and have the same number
    # of loop iterations (forward passes), otherwise NCCL will deadlock.
    #
    # Solution: Select from cached files only, and use deterministic selection based on
    # global_step so all ranks select the same file index. If caches differ, ranks will
    # coordinate via all_reduce.

    # Get cached files from H5Cache
    cached_files = train_ds.h5cache.get_loaded_files() if hasattr(train_ds, 'h5cache') and train_ds.h5cache is not None else []

    if len(cached_files) == 0:
        # No files cached yet - skip scheduled sampling
        if rank == 0:
            print(f"[SCHED_SAMP] step {global_step}: No cached files available yet, skipping", flush=True)
        return None

    # Deterministic file selection based on global_step (all ranks use same index)
    file_idx = global_step % len(cached_files)
    file_path = cached_files[file_idx]

    # Create a seeded rng for other random decisions within this function
    rng = random.Random(global_step)

    # Get trajectory data - should succeed since we selected from cached files
    # Returns None if file is not in cache (shouldn't happen but handle it)
    traj = train_ds.get_trajectory(file_path=file_path)
    has_file = 1.0 if traj is not None else 0.0

    # Use all_reduce to check if ALL ranks have the file
    # MIN of all values: if any rank has 0, result is 0
    # This handles the case where different ranks have different cached files
    if dist.is_initialized():
        has_file_tensor = torch.tensor([has_file], device=device, dtype=torch.float32)
        dist.all_reduce(has_file_tensor, op=dist.ReduceOp.MIN)
        all_have_file = has_file_tensor.item() > 0.5
    else:
        all_have_file = has_file > 0.5

    if not all_have_file:
        # At least one rank doesn't have the file - ALL ranks skip together
        if rank == 0:
            print(f"[SCHED_SAMP] step {global_step}: Cache mismatch across ranks, skipping", flush=True)
        return None

    # Load trajectory data
    static = traj["static"].to(device)  # [C_static, Z, Y, X]
    params = traj["params"].to(device)  # [26]
    grid_all = traj["grid_all"].to(device)  # [T, 3, Z, Y, X] - normalized inputs
    deltas_all = traj["deltas_all"].to(device)  # [T-1, 3, Z, Y, X] - normalized target deltas
    valid_mask = traj["valid_mask"].to(device)  # [Z, Y, X]
    T = traj["T"]

    # CRITICAL for DDP: Synchronize T across all ranks to ensure same loop iterations
    # Use all_reduce with MIN to get minimum T across ranks (handles any edge cases)
    if dist.is_initialized():
        T_tensor = torch.tensor([float(T)], device=device, dtype=torch.float32)
        dist.all_reduce(T_tensor, op=dist.ReduceOp.MIN)
        T = int(T_tensor.item())

    # If T is too small for rollout, skip (all ranks have same T after sync)
    if T < 3:
        return None

    # Pad if needed
    Z, Y, X = static.shape[1], static.shape[2], static.shape[3]
    if fixed_pad_size is not None:
        pad_z = fixed_pad_size[0] - Z
        pad_y = fixed_pad_size[1] - Y
        pad_x = fixed_pad_size[2] - X
        padding = (0, pad_x, 0, pad_y, 0, pad_z)
        static = torch.nn.functional.pad(static, padding, mode='constant', value=0)
        grid_all = torch.nn.functional.pad(grid_all, padding, mode='constant', value=0)
        deltas_all = torch.nn.functional.pad(deltas_all, padding, mode='constant', value=0)
        valid_mask = torch.nn.functional.pad(valid_mask, padding, mode='constant', value=0)

    # Stats for converting between normalized delta and raw delta
    delta_std = torch.tensor(stats["std"]["grid_output_delta"], device=device, dtype=torch.float32).clamp(min=1e-6)
    input_mean = torch.tensor(stats["mean"]["grid_input_absolute"], device=device, dtype=torch.float32)
    input_std = torch.tensor(stats["std"]["grid_input_absolute"], device=device, dtype=torch.float32).clamp(min=1e-6)

    # Track current "belief" state (normalized) - initially use ground truth
    # belief_state[c] is the model's current belief about channel c at time t
    # Shape: [3, Z, Y, X]
    belief_state = grid_all[1].clone().detach()  # Start with ground truth at t=1

    # Track losses for reporting (detached, not used for gradients)
    total_loss_p = 0.0
    total_loss_T = 0.0
    total_loss_wept = 0.0
    num_steps = 0

    # Pick a random starting point and do limited rollout to avoid OOM
    # Use seeded rng for DDP sync
    max_t = min(T - 1, max_rollout_steps + 1)
    start_t = rng.randint(1, max(1, T - max_rollout_steps - 1))
    end_t = min(start_t + max_rollout_steps, T - 1)

    # CRITICAL for DDP: Broadcast loop bounds from rank 0 to ensure identical iterations
    if dist.is_initialized():
        loop_bounds = torch.tensor([float(start_t), float(end_t)], device=device, dtype=torch.float32)
        dist.broadcast(loop_bounds, src=0)
        start_t = int(loop_bounds[0].item())
        end_t = int(loop_bounds[1].item())

    # Initialize belief state at start_t (detached, no gradient yet)
    belief_state = grid_all[start_t].clone().detach()

    # Rollout from start_t to end_t
    for t in range(start_t, end_t):
        # Get ground truth states at t-1 and t
        gt_tm1 = grid_all[t - 1]  # [3, Z, Y, X]
        gt_t = grid_all[t]  # [3, Z, Y, X]
        gt_delta = deltas_all[t]  # [3, Z, Y, X] - target delta for t -> t+1

        # Decide whether to use model prediction or ground truth for current state
        # Use seeded rng for DDP sync
        if t > start_t and rng.random() < sched_prob:
            # Use model's belief for current state (on-policy)
            current_state = belief_state.detach()  # Detach to prevent gradient through belief
        else:
            # Use ground truth (teacher forcing)
            current_state = gt_t.detach()

        # For t-1, always use ground truth
        prev_state = gt_tm1.detach()

        # Build input tensor: [static(8), P_{t-1}, T_{t-1}, WEPT_{t-1}, P_t, T_t, WEPT_t]
        x_grid = torch.cat([
            static,  # [8, Z, Y, X]
            prev_state[0:1],  # P_{t-1}
            prev_state[1:2],  # T_{t-1}
            prev_state[2:3],  # WEPT_{t-1}
            current_state[0:1],  # P_t
            current_state[1:2],  # T_t
            current_state[2:3],  # WEPT_t
        ], dim=0).unsqueeze(0)  # [1, 14, Z, Y, X]

        # Forward pass with gradient
        with torch.cuda.amp.autocast(enabled=use_amp):
            grid_pred, _ = model(x_grid, params.unsqueeze(0))  # [1, 3, Z, Y, X]
            grid_pred = grid_pred.squeeze(0)  # [3, Z, Y, X]

            # Compute loss against ground truth delta
            se_p = (grid_pred[0] - gt_delta[0]) ** 2
            se_T = (grid_pred[1] - gt_delta[1]) ** 2
            se_wept = (grid_pred[2] - gt_delta[2]) ** 2

            loss_p = (se_p * valid_mask).sum() / valid_mask.sum().clamp(min=1)
            loss_T = (se_T * valid_mask).sum() / valid_mask.sum().clamp(min=1)
            loss_wept = (se_wept * valid_mask).sum() / valid_mask.sum().clamp(min=1)

            # FIX: Clamp individual losses to prevent destabilizing spikes from bad rollouts
            # Max reasonable MSE: ~100 (RMSE ~10 in normalized space)
            loss_p = loss_p.clamp(max=100.0)
            loss_T = loss_T.clamp(max=100.0)
            loss_wept = loss_wept.clamp(max=100.0)

            step_loss = loss_p + loss_T + loss_wept

        # Backward pass per step (no BPTT - each step independently)
        if scaler is not None:
            scaler.scale(step_loss).backward()
        else:
            step_loss.backward()

        # Track losses for reporting (detached)
        total_loss_p += loss_p.detach().item()
        total_loss_T += loss_T.detach().item()
        total_loss_wept += loss_wept.detach().item()
        num_steps += 1

        # Update belief state for next iteration (no gradient needed)
        with torch.no_grad():
            # Convert predicted normalized delta to raw delta
            pred_delta_raw = grid_pred.detach() * delta_std.view(3, 1, 1, 1)

            # Convert current state from normalized to raw
            current_raw = current_state * input_std.view(3, 1, 1, 1) + input_mean.view(3, 1, 1, 1)

            # Add predicted delta to get predicted next state (raw)
            next_raw = current_raw + pred_delta_raw

            # Normalize back to input space
            belief_state = (next_raw - input_mean.view(3, 1, 1, 1)) / input_std.view(3, 1, 1, 1)

    # Return average loss per step (as floats)
    num_steps = max(1, num_steps)
    avg_loss = (total_loss_p + total_loss_T + total_loss_wept) / num_steps
    return avg_loss, num_steps, total_loss_p / num_steps, total_loss_T / num_steps, total_loss_wept / num_steps


def main():
    ap = argparse.ArgumentParser(description="Train voxel autoregressive model")

    # Data
    ap.add_argument("--prepped_root", type=str, default=None, help="Root with stats.json and indices (for prepped data mode)")
    ap.add_argument("--train_index", type=str, default=None)
    ap.add_argument("--test_index", type=str, default=None)
    ap.add_argument("--raw_h5_mode", action="store_true", help="Use raw h5 files directly (no preprocessing)")
    ap.add_argument("--raw_h5_dir", type=str, default="/workspace/all_oak_data/h5s_v2.5_data", help="Directory with raw v2.5_*.h5 files")
    ap.add_argument("--stats_path", type=str, default=None, help="Path to stats.json (required for raw_h5_mode)")

    # Dataloader
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--pin_memory", action="store_true")

    # Model
    ap.add_argument("--receptive_field_radius", type=int, default=2, help="Receptive field radius (kernel size = 2*r+1)")
    ap.add_argument("--base_channels", type=int, default=64)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--use_param_broadcast", action="store_true")
    ap.add_argument("--use_checkpoint", action="store_true", help="Use gradient checkpointing to reduce memory.. NOT resume from checkpoint")
    ap.add_argument("--baseline_linreg", action="store_true", help="Use simple linear regression baseline (y=Wx+b) instead of VoxelAutoRegressor")

    # Training
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_steps", type=int, default=10)
    ap.add_argument("--max_steps", type=int, default=1000000)
    ap.add_argument("--accum_steps", type=int, default=10)
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--eval_every", type=int, default=100)
    ap.add_argument("--ckpt_every", type=int, default=1000)
    ap.add_argument("--rollout_timeout", type=int, default=600, help="Timeout in seconds for rollout eval (default 600s = 10 min)")
    ap.add_argument("--skip_rollout_eval", action="store_true", help="Skip rollout evaluation during training (avoids NCCL timeout issues)")
    ap.add_argument("--save_dir", type=str, default="./checkpoints")
    ap.add_argument("--seed", type=int, default=42)

    # Augmentations
    ap.add_argument("--aug_xy_rot", action="store_true")
    ap.add_argument("--aug_flip", action="store_true")
    ap.add_argument("--noise_std", type=float, default=0.0)
    ap.add_argument("--fixed_pad_size", type=str, default=None, help="Fixed padding size Z,Y,X (e.g., '60,64,64') to prevent memory fragmentation")

    # Training options
    ap.add_argument("--use_amp", action="store_true")
    ap.add_argument("--use_wandb", action="store_true")
    ap.add_argument("--wandb_project", type=str, default="voxel-ode")
    ap.add_argument("--wandb_run_name", type=str, default=None)
    ap.add_argument("--resume_from", type=str, default=None, help="Path to checkpoint to resume from")
    ap.add_argument("--predict_grid_only", action="store_true", help="Only predict pressure/temperature (no WEPT in v2.5, no scalar fields)")
    ap.add_argument("--predict_residuals", action="store_true", default=False, help="Predict residuals (deltas) instead of absolute values (recommended for v2.5)")
    ap.add_argument("--no_predict_residuals", dest="predict_residuals", action="store_false", help="Predict absolute values instead of residuals")
    ap.add_argument("--use_torch_compile", action="store_true", help="Use torch.compile for model speedup (PyTorch 2.0+)")

    # Scheduled sampling (acc5-based, not step-based)
    ap.add_argument("--scheduled_sampling", action="store_true", help="Enable scheduled sampling (mix model predictions with ground truth)")
    ap.add_argument("--sched_samp_start_acc5", type=float, default=0.95, help="Acc5 threshold to start scheduled sampling (e.g., 0.95 = 95%%)")
    ap.add_argument("--sched_samp_end_acc5", type=float, default=0.98, help="Acc5 threshold for full scheduled sampling (e.g., 0.98 = 98%%)")
    ap.add_argument("--sched_samp_final_prob", type=float, default=0.3, help="Final probability of using model predictions at high acc5")
    ap.add_argument("--sched_samp_window", type=int, default=10, help="Number of recent eval steps to average acc5 over")
    ap.add_argument("--wept_loss_weight", type=float, default=1.0, help="Weight for WEPT loss (set to 0 to ignore WEPT)")
    ap.add_argument("--acc_abs_p", type=float, default=5.0, help="Absolute accuracy threshold for pressure in bar (default 5 bar)")
    ap.add_argument("--acc_abs_t", type=float, default=5.0, help="Absolute accuracy threshold for temperature in C (default 5 C)")
    ap.add_argument("--acc_abs_wept", type=float, default=1e10, help="Absolute accuracy threshold for WEPT in J (default 1e10 J)")
    ap.add_argument("--cache_max_files", type=int, default=150, help="Max raw H5 files to keep resident in RAM")
    ap.add_argument("--cache_initial_files", type=int, default=1, help="Number of raw H5 files to preload synchronously before training starts")
    ap.add_argument("--cache_prefetch_steps", type=int, default=5, help="Optimizer steps between asynchronous raw H5 loads (<=0 disables)")
    ap.add_argument("--cache_load_workers", type=int, default=1, help="Background threads for raw H5 cache loading")

    args = ap.parse_args()
    cache_prefetch_steps = args.cache_prefetch_steps if args.cache_prefetch_steps > 0 else None

    print(args)

    os.makedirs(args.save_dir, exist_ok=True)

    # Validate arguments
    if args.raw_h5_mode:
        if args.stats_path is None:
            raise ValueError("--stats_path is required when using --raw_h5_mode")
        if not os.path.exists(args.stats_path):
            raise ValueError(f"Stats file not found: {args.stats_path}")
        ddp_print(f"RAW H5 MODE: Training directly from {args.raw_h5_dir}")
        ddp_print(f"Using stats from: {args.stats_path}")
    else:
        if args.prepped_root is None:
            raise ValueError("--prepped_root is required when not using --raw_h5_mode")
        ddp_print(f"PREPPED DATA MODE: Training from {args.prepped_root}")

    # Initialize DDP
    use_ddp = init_distributed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    set_seed(args.seed + (dist.get_rank() if dist.is_initialized() else 0))

    # Load stats and create datasets based on mode
    if args.raw_h5_mode:
        # Raw H5 mode: load stats from specified path
        with open(args.stats_path, "r") as f:
            stats = json.load(f)

        # Create RawH5Dataset instances
        train_ds = RawH5Dataset(
            raw_h5_dir=args.raw_h5_dir,
            stats_path=args.stats_path,
            split="train",
            augment=True,
            aug_xy_rot=args.aug_xy_rot,
            aug_flip=args.aug_flip,
            noise_std=args.noise_std,
            predict_residuals=args.predict_residuals,
            files_in_memory=args.cache_max_files,
            initial_preload=args.cache_initial_files,
            cache_prefetch_steps=cache_prefetch_steps,
            async_prefetch=True,
            cache_load_workers=args.cache_load_workers,
        )
        test_ds = RawH5Dataset(
            raw_h5_dir=args.raw_h5_dir,
            stats_path=args.stats_path,
            split="test",
            augment=False,
            aug_xy_rot=False,
            aug_flip=False,
            noise_std=0.0,
            predict_residuals=args.predict_residuals,
            files_in_memory=min(args.cache_max_files, 32),
            initial_preload=min(args.cache_initial_files, 5),
            cache_prefetch_steps=None,
            async_prefetch=False,
        )
    else:
        # Prepped data mode: original behavior
        stats = load_stats(args.prepped_root)
        train_index = args.train_index or os.path.join(args.prepped_root, "index_train.jsonl")
        test_index = args.test_index or os.path.join(args.prepped_root, "index_test.jsonl")

        train_ds = VoxelARIndexDataset(
            index_path=train_index,
            stats_path=os.path.join(args.prepped_root, "stats.json"),
            augment=True,
            aug_xy_rot=args.aug_xy_rot,
            aug_flip=args.aug_flip,
            noise_std=args.noise_std,
            use_param_broadcast=args.use_param_broadcast,
            use_params_as_condition=True,
            predict_grid_only=args.predict_grid_only,
            predict_residuals=args.predict_residuals,
        )
        test_ds = VoxelARIndexDataset(
            index_path=test_index,
            stats_path=os.path.join(args.prepped_root, "stats.json"),
            augment=False,
            aug_xy_rot=False,
            aug_flip=False,
            noise_std=0.0,
            use_param_broadcast=args.use_param_broadcast,
            use_params_as_condition=True,
            predict_grid_only=args.predict_grid_only,
            predict_residuals=args.predict_residuals,
        )

    # Samplers - always make them DDP aware
    if use_ddp:
        # For raw H5 mode, use DDP-aware file-grouped sampler for better file locality
        if args.raw_h5_mode:
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            train_sampler = DDPFileGroupedBatchSampler(
                dataset_size=len(train_ds),
                batch_size=args.batch_size,
                accum_steps=args.accum_steps,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                drop_last=True,
                seed=args.seed,
            )
            ddp_print(f"Using DDPFileGroupedBatchSampler with accum_steps={args.accum_steps}")
            ddp_print("This groups consecutive batches to reuse file handles and reduce network I/O")
            # Test sampler can remain simple - file reuse less critical during eval
            test_sampler = DistributedSampler(test_ds, shuffle=False, drop_last=False)
        else:
            train_sampler = DistributedSampler(train_ds, shuffle=True, drop_last=True)
            test_sampler = DistributedSampler(test_ds, shuffle=False, drop_last=False)
    else:
        train_sampler = None
        test_sampler = None

    # Parse fixed padding size if provided
    fixed_size = None
    if args.fixed_pad_size:
        fixed_size = tuple(map(int, args.fixed_pad_size.split(',')))
        ddp_print(f"Using fixed padding size: Z={fixed_size[0]}, Y={fixed_size[1]}, X={fixed_size[2]}")
        ddp_print("This prevents memory fragmentation from variable batch sizes")
        collate_fn = partial(pad_collate_fn, fixed_size=fixed_size)
    else:
        collate_fn = pad_collate_fn

    # Loaders - handle batch sampler vs regular sampler
    if args.raw_h5_mode and use_ddp:
        # Using batch sampler - pass to batch_sampler parameter
        train_loader = DataLoader(
            train_ds,
            batch_sampler=train_sampler,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            persistent_workers=(args.num_workers > 0),
            collate_fn=collate_fn,
        )
    else:
        # Using regular sampler - pass to sampler parameter with batch_size
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            sampler=train_sampler,
            shuffle=(train_sampler is None),
            drop_last=True,
            persistent_workers=(args.num_workers > 0),
            collate_fn=collate_fn,
        )

    test_loader = DataLoader(
        test_ds,
        batch_size=max(1, args.batch_size // 2),
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        sampler=test_sampler,
        shuffle=False,
        drop_last=False,
        persistent_workers=(args.num_workers > 0),
        collate_fn=collate_fn,
    )

    # Model
    C_static = len(stats["static_channels"])
    data_format = stats.get("data_format", "v2.4")
    num_grid_channels = len(stats["grid_channels"])

    # Raw H5 mode is always v2.5 format with 3 output channels
    if args.raw_h5_mode:
        data_format = "v2.5"
        grid_out_channels = 3
        # Raw H5 always uses full grids (no grid_only mode)
        if args.predict_grid_only:
            ddp_print("WARNING: predict_grid_only is not supported in raw_h5_mode, ignoring")
            args.predict_grid_only = False

    # Compute input channels for 2-timestep context
    # Input structure: [static, P_{t-1}, T_{t-1}, (WEPT_{t-1}), P_t, T_t, (WEPT_t)]
    # v2.5 not grid_only: 8 + 3 + 3 = 14 channels
    # v2.4 or v2.5 grid_only: 8 + 2 + 2 = 12 channels
    in_channels = C_static  # Static features

    if data_format == "v2.5" and not args.predict_grid_only:
        # v2.5: P, T, WEPT at both t-1 and t
        in_channels += 3 + 3  # 3 channels from t-1, 3 from t
    else:
        # v2.4 or grid_only: P, T at both t-1 and t
        in_channels += 2 + 2  # 2 channels from t-1, 2 from t

    if args.use_param_broadcast:
        in_channels += 26

    # Compute grid output channels (if not already set by raw_h5_mode)
    # v2.4: always 2 (P, T)
    # v2.5: 2 if predict_grid_only, 3 if not (P, T, WEPT)
    if not args.raw_h5_mode:
        if data_format == "v2.5" and not args.predict_grid_only:
            grid_out_channels = 3
        else:
            grid_out_channels = 2

    ddp_print(f"Data format: {data_format}")
    ddp_print(f"Creating model with in_channels={in_channels}, grid_out_channels={grid_out_channels}, base_channels={args.base_channels}, depth={args.depth}")
    if args.predict_grid_only:
        ddp_print("Running in GRID ONLY mode - only predicting pressure and temperature")
    elif data_format == "v2.5":
        ddp_print("v2.5 mode - predicting pressure, temperature, and WEPT")

    try:
        if args.baseline_linreg:
            model = LinearBaseline(
                in_channels=in_channels,
                grid_out_channels=grid_out_channels,
                scalar_out_dim=5,
                cond_params_dim=26,
                use_param_broadcast=args.use_param_broadcast,
            )
            ddp_print(f"LINEAR BASELINE model created with {sum(p.numel() for p in model.parameters())} parameters")
        else:
            model = VoxelAutoRegressor(
                in_channels=in_channels,
                base_channels=args.base_channels,
                depth=args.depth,
                r=args.receptive_field_radius,
                cond_params_dim=26,
                use_param_broadcast=args.use_param_broadcast,
                grid_out_channels=grid_out_channels,
                scalar_out_dim=5,
                use_checkpoint=args.use_checkpoint,
            )
            ddp_print(f"Model created successfully with {sum(p.numel() for p in model.parameters())} parameters")
            if args.use_checkpoint:
                ddp_print("Gradient checkpointing ENABLED - trading compute for memory")

        model = model.to(device)
        ddp_print(f"Model moved to device {device}")
    except Exception as e:
        print(f"[RANK {dist.get_rank() if dist.is_initialized() else 0}] ERROR creating model: {e}")
        raise

    # Apply torch.compile if requested (before DDP wrapping for best performance)
    if args.use_torch_compile:
        ddp_print("Applying torch.compile to model for speedup...")
        try:
            model = torch.compile(model, mode="default")
            ddp_print("torch.compile applied successfully (mode='default')")
        except Exception as e:
            ddp_print(f"WARNING: torch.compile failed: {e}")
            ddp_print("Continuing without torch.compile...")

    # Synchronize before wrapping in DDP
    if use_ddp:
        dist.barrier()
        ddp_print("All ranks synchronized before DDP wrapping")
        # When scheduled sampling is enabled, we need find_unused_parameters=True because
        # the rollout forward passes only use grid outputs, not scalar head outputs
        find_unused = args.scheduled_sampling
        model = DDP(model, device_ids=[int(os.environ["LOCAL_RANK"])], find_unused_parameters=find_unused)
        ddp_print(f"Model wrapped in DDP successfully (find_unused_parameters={find_unused})")

    # Optimizer & Scheduler
    use_lbfgs = args.baseline_linreg  # Use LBFGS for linear regression baseline
    if use_lbfgs:
        opt = torch.optim.LBFGS(model.parameters(), lr=args.lr, max_iter=20, history_size=10, line_search_fn="strong_wolfe")
        ddp_print(f"Using LBFGS optimizer for linear regression baseline (lr={args.lr})")
        sched = None  # LBFGS doesn't need a scheduler typically
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))
        sched = WarmupCosine(opt, warmup_steps=args.warmup_steps, max_steps=args.max_steps)

    # AMP
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)

    # Resume from checkpoint if specified
    start_step = 0
    if args.resume_from:
        if os.path.exists(args.resume_from):
            ddp_print(f"Loading checkpoint from {args.resume_from}")
            checkpoint = torch.load(args.resume_from, map_location=device)

            # Load model state
            if use_ddp:
                model.module.load_state_dict(checkpoint["model"])
            else:
                model.load_state_dict(checkpoint["model"])

            # Load optimizer, scheduler, scaler state if available (old checkpoints may not have these)
            if "optimizer" in checkpoint:
                opt.load_state_dict(checkpoint["optimizer"])
                ddp_print("Loaded optimizer state")
            else:
                ddp_print("WARNING: Optimizer state not in checkpoint, using fresh optimizer")

            if "scheduler" in checkpoint:
                sched.load_state_dict(checkpoint["scheduler"])
                ddp_print("Loaded scheduler state")
            else:
                ddp_print("WARNING: Scheduler state not in checkpoint, using fresh scheduler")

            if "scaler" in checkpoint:
                scaler.load_state_dict(checkpoint["scaler"])
                ddp_print("Loaded scaler state")
            else:
                ddp_print("WARNING: Scaler state not in checkpoint, using fresh scaler")

            start_step = checkpoint["step"]
            ddp_print(f"Resumed from step {start_step}")
        else:
            ddp_print(f"WARNING: Checkpoint {args.resume_from} not found, starting from scratch")

    # W&B
    run = None
    if args.use_wandb:
        if not dist.is_initialized() or dist.get_rank() == 0:
            import wandb
            run = wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))

    # Evaluate at step 0 (skip if resuming) - FULL evaluation with rollout for baseline
    if start_step == 0:
        if use_ddp and isinstance(test_loader.sampler, DistributedSampler):
            test_loader.sampler.set_epoch(0)

        eval_start_time = time.time()

        # 1-step eval (teacher forcing) - use same max_batches as training loop (50)
        ddp_print(f"\n[EVAL step 0] Starting 1-step evaluation (BASELINE)...")
        onestep_start = time.time()
        metrics0 = evaluate(model, test_loader, device, stats, stats["log1p_flags"]["scalar"], max_batches=50, predict_grid_only=args.predict_grid_only, grid_out_channels=grid_out_channels,
                            acc_abs_p=args.acc_abs_p, acc_abs_t=args.acc_abs_t, acc_abs_wept=args.acc_abs_wept)
        onestep_time = time.time() - onestep_start

        # Full rollout eval (NO teacher forcing) - only on rank 0
        # Can be skipped with --skip_rollout_eval to avoid NCCL timeout issues
        rollout_metrics0 = {}
        if (not dist.is_initialized() or dist.get_rank() == 0) and args.raw_h5_mode and not args.skip_rollout_eval:
            ddp_print(f"  Starting full rollout evaluation on test files (timeout={args.rollout_timeout}s)...")
            rollout_start = time.time()
            fixed_pad = None
            if args.fixed_pad_size:
                fixed_pad = tuple(int(x) for x in args.fixed_pad_size.split(","))
            try:
                # Set timeout using signal alarm
                old_handler = signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(args.rollout_timeout)
                rollout_metrics0 = evaluate_rollout(
                    model, args.raw_h5_dir, stats, device,
                    fixed_pad_size=fixed_pad, max_steps=29,
                    acc_abs_p=args.acc_abs_p, acc_abs_t=args.acc_abs_t, acc_abs_wept=args.acc_abs_wept
                )
                signal.alarm(0)  # Cancel the alarm
                signal.signal(signal.SIGALRM, old_handler)  # Restore old handler
                rollout_time = time.time() - rollout_start
                ddp_print(f"  Rollout eval completed in {rollout_time:.1f}s")
            except TimeoutError:
                signal.alarm(0)  # Cancel the alarm
                ddp_print(f"  WARNING: Rollout eval timed out after {args.rollout_timeout}s - skipping this eval")
                rollout_metrics0 = {}  # Empty metrics on timeout
            except Exception as e:
                signal.alarm(0)  # Cancel the alarm
                ddp_print(f"  WARNING: Rollout eval failed with error: {e} - skipping this eval")
                rollout_metrics0 = {}  # Empty metrics on error

        # Compute validation score
        if grid_out_channels == 3:
            val_score0 = metrics0["mse/grid_pressure"] + metrics0["mse/grid_temperature"] + metrics0["mse/grid_wept"]
        elif args.predict_grid_only:
            val_score0 = metrics0["mse/grid_pressure"] + metrics0["mse/grid_temperature"]
        else:
            val_score0 = metrics0["mse/grid_pressure"] + metrics0["mse/grid_temperature"] + sum(
                metrics0[f"mse/{name}"] for name in stats["scalar_channels"]
            ) / len(stats["scalar_channels"])

        eval_total_time = time.time() - eval_start_time

        # Log to wandb
        if run and (not dist.is_initialized() or dist.get_rank() == 0):
            for k, v in metrics0.items():
                run.log({f"test/{k}": v, "step": 0})
            run.log({"test/val_score": val_score0, "step": 0})
            for k, v in rollout_metrics0.items():
                run.log({f"test/{k}": v, "step": 0})

        # Comprehensive stdout logging (same format as training loop eval)
        if not dist.is_initialized() or dist.get_rank() == 0:
            eval_msg = f"[EVAL step 0] COMPLETED in {eval_total_time:.1f}s (1-step: {onestep_time:.1f}s) - BASELINE"
            eval_msg += f"\n  val_score={val_score0:.6e}"
            eval_msg += f"\n  [1-STEP] MSE - P:{metrics0['mse/grid_pressure']:.6e} T:{metrics0['mse/grid_temperature']:.6e}"
            if grid_out_channels == 3:
                eval_msg += f" WEPT:{metrics0['mse/grid_wept']:.6e}"
            eval_msg += f"\n  [1-STEP] ACC_ABS (P<{args.acc_abs_p}bar, T<{args.acc_abs_t}C) - P:{metrics0['acc5/grid_pressure']:.4f} T:{metrics0['acc5/grid_temperature']:.4f}"
            if grid_out_channels == 3:
                eval_msg += f" WEPT:{metrics0['acc5/grid_wept']:.4f}"

            # Log rollout metrics if available
            if rollout_metrics0:
                eval_msg += f"\n  [ROLLOUT 29-step] MSE - P:{rollout_metrics0['rollout/mse_p']:.6e} T:{rollout_metrics0['rollout/mse_t']:.6e} WEPT:{rollout_metrics0['rollout/mse_w']:.6e}"
                eval_msg += f"\n  [ROLLOUT 29-step] ACC_ABS - P:{rollout_metrics0['rollout/acc5_p']:.4f} T:{rollout_metrics0['rollout/acc5_t']:.4f} WEPT:{rollout_metrics0['rollout/acc5_w']:.4f}"
                eval_msg += f" (files: {rollout_metrics0.get('rollout/files_evaluated', 0)})"

            ddp_print(eval_msg)

        # Barrier after step 0 eval to ensure all ranks sync before training starts
        # Rank 0 does extra rollout work, other ranks wait here
        if dist.is_initialized():
            dist.barrier()

    # Training loop
    global_step = start_step
    best_val = float("inf")
    t0 = time.time()
    last_log_time = t0  # For accurate time/step measurement
    last_log_step = start_step  # Track which step we last logged at

    # Track acc5 history for scheduled sampling
    acc5_history = []  # List of (step, avg_acc5) tuples
    current_sched_samp_prob = 0.0  # Current scheduled sampling probability

    ddp_print(f"Starting training for {args.max_steps} steps...")
    if args.scheduled_sampling:
        ddp_print(f"Scheduled sampling ENABLED:")
        ddp_print(f"  - Start at avg_acc5 >= {args.sched_samp_start_acc5*100:.0f}%")
        ddp_print(f"  - Full prob at avg_acc5 >= {args.sched_samp_end_acc5*100:.0f}%")
        ddp_print(f"  - Final prob: {args.sched_samp_final_prob*100:.0f}%")
        ddp_print(f"  - Rolling window: {args.sched_samp_window} eval steps")

    while global_step < args.max_steps:
        if use_ddp and isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(global_step // max(1, len(train_loader)) + 1)

        for it, batch in enumerate(train_loader):
            model.train()

            # CRITICAL for DDP: Use skip_batch flag instead of continue to avoid desync
            # All ranks must iterate through the same number of batches and hit the same barriers
            skip_batch = False

            # Scheduled sampling: use DETERMINISTIC decision based on global_step
            # so all DDP ranks make the same choice (essential for gradient sync)
            import random
            sched_rng = random.Random(global_step + 12345)  # Different seed than internal rng
            use_sched_samp_this_step = (
                args.scheduled_sampling and
                current_sched_samp_prob > 0 and
                sched_rng.random() < current_sched_samp_prob and
                args.raw_h5_mode and
                hasattr(train_ds, 'get_trajectory')
            )

            if use_sched_samp_this_step and not skip_batch:
                # On-policy rollout training - log when this happens for debugging
                if global_step % args.log_every == 0:
                    ddp_print(f"[SCHED_SAMP] step {global_step}: Using scheduled sampling with prob={current_sched_samp_prob:.3f}")
                # Parse fixed_pad_size if specified
                fixed_pad = None
                if args.fixed_pad_size:
                    fixed_pad = tuple(int(x) for x in args.fixed_pad_size.split(","))

                # Perform rollout with model predictions mixed in
                # Note: scheduled_sampling_step now does backward internally
                # IMPORTANT: Pass the DDP-wrapped model, not model.module, for gradient sync
                sched_result = scheduled_sampling_step(
                    model=model,  # Use DDP-wrapped model for gradient synchronization
                    train_ds=train_ds,
                    device=device,
                    sched_prob=current_sched_samp_prob,
                    stats=stats,
                    use_amp=args.use_amp,
                    fixed_pad_size=fixed_pad,
                    scaler=scaler,
                    max_rollout_steps=25,  # Train on longer rollouts (eval is 29 steps)
                    global_step=global_step  # For seeding random decisions
                )

                # Handle case where scheduled sampling was skipped (file not in cache)
                # All ranks will skip together since they select the same file_path
                if sched_result is None:
                    # Fall back to normal training instead of skipping
                    # This prevents loss=0 steps when scheduled sampling can't proceed
                    if global_step % args.log_every == 0:
                        ddp_print(f"[SCHED_SAMP] step {global_step}: Skipped (cache issue), falling back to normal training")
                    use_sched_samp_this_step = False  # Fall through to normal training
                else:
                    # Scheduled sampling succeeded!
                    if global_step % args.log_every == 0:
                        ddp_print(f"[SCHED_SAMP] step {global_step}: SUCCESS - rollout training completed")
                    loss, num_rollout_steps, loss_grid_p, loss_grid_T, loss_grid_wept = sched_result
                    loss_scalar = 0.0

                    # Check for NaN loss (loss is now a float, not tensor)
                    import math
                    if math.isnan(loss) or math.isinf(loss):
                        ddp_print(f"WARNING: NaN/inf loss in sched_samp at step {global_step}, skipping")
                        opt.zero_grad(set_to_none=True)
                        skip_batch = True

                # Gradients already accumulated in scheduled_sampling_step, no need for backward here

            if not use_sched_samp_this_step and not skip_batch:
                # Normal batch training (teacher forcing)
                x_grid = batch["x_grid"].to(device, non_blocking=True)
                params = batch["params"].to(device, non_blocking=True)
                y_grid = batch["y_grid"].to(device, non_blocking=True)
                y_scalar = batch["y_scalar"].to(device, non_blocking=True)
                valid_mask = batch["valid_mask"].to(device, non_blocking=True)  # [B, Z, Y, X]

                # Check for NaN/inf in inputs - set skip_batch flag instead of continue
                if torch.isnan(x_grid).any() or torch.isinf(x_grid).any():
                    ddp_print(f"WARNING: NaN/inf detected in input at step {global_step}, skipping batch")
                    skip_batch = True
                if torch.isnan(y_grid).any() or torch.isinf(y_grid).any():
                    ddp_print(f"WARNING: NaN/inf detected in target at step {global_step}, skipping batch")
                    skip_batch = True

            # Only do forward/backward if not skipping
            if not skip_batch and not use_sched_samp_this_step:
                # Avoid gradient allreduce on every microbatch when accumulating
                if use_ddp and args.accum_steps > 1 and ((it + 1) % args.accum_steps != 0):
                    sync_context = model.no_sync
                else:
                    sync_context = nullcontext

                with sync_context():
                    with torch.cuda.amp.autocast(enabled=args.use_amp):
                        grid_pred, scalar_pred = model(x_grid, params)

                        # Compute masked MSE losses in WHITENED space (only on valid cells, ignore -999)
                        se_p = (grid_pred[:, 0] - y_grid[:, 0]) ** 2  # [B, Z, Y, X]
                        se_T = (grid_pred[:, 1] - y_grid[:, 1]) ** 2  # [B, Z, Y, X]

                        # Apply mask and normalize by number of valid cells
                        # NOTE: valid_mask already filters out -999 values, no need for additional pressure mask
                        # The old code incorrectly checked y_grid[:, 0] != 0 in WHITENED space,
                        # which masked out pressures near the mean (184 bar) incorrectly
                        loss_grid_p = (se_p * valid_mask).sum() / valid_mask.sum().clamp(min=1)
                        loss_grid_T = (se_T * valid_mask).sum() / valid_mask.sum().clamp(min=1)

                        # Add WEPT loss if in v2.5 and not grid_only
                        if grid_out_channels == 3:
                            se_wept = (grid_pred[:, 2] - y_grid[:, 2]) ** 2
                            loss_grid_wept = (se_wept * valid_mask).sum() / valid_mask.sum().clamp(min=1)
                        else:
                            loss_grid_wept = torch.tensor(0.0, device=device)

                        # Scalars:
                        # - In v2.5 / raw_h5_mode we do not supervise scalars, but we still
                        #   tie scalar_pred into the graph with zero weight so DDP does not
                        #   see scalar_head parameters as unused.
                        if args.predict_grid_only:
                            loss_scalar = (scalar_pred * 0.0).sum()
                        else:
                            if grid_out_channels == 3:
                                # v2.5 mode: grid fields only (P, T, WEPT), no scalar supervision
                                loss_scalar = (scalar_pred * 0.0).sum()
                            else:
                                # v2.4 mode: grid fields (P, T) + scalars
                                loss_scalar = torch.mean((scalar_pred - y_scalar) ** 2)

                        # Total loss always includes P, T, WEPT (if present) and scalar term
                        loss = loss_grid_p + loss_grid_T + args.wept_loss_weight * loss_grid_wept + loss_scalar

                    # Check for NaN loss - set skip_batch flag instead of continue
                    if torch.isnan(loss) or torch.isinf(loss):
                        ddp_print(f"WARNING: NaN/inf loss at step {global_step}, skipping batch")
                        ddp_print(f"  loss_grid_p: {loss_grid_p.item()}, loss_grid_T: {loss_grid_T.item()}, loss_scalar: {loss_scalar.item()}")
                        opt.zero_grad(set_to_none=True)
                        skip_batch = True

                    if not skip_batch:
                        if use_lbfgs:
                            # LBFGS: no scaler, direct backward
                            (loss / args.accum_steps).backward()
                        else:
                            scaler.scale(loss / args.accum_steps).backward()

            if ((it + 1) % args.accum_steps) == 0:
                # Check for NaN gradients - use flag instead of continue to avoid DDP desync
                # All ranks must hit the same barriers regardless of NaN or skip_batch
                skip_update = skip_batch  # If batch was skipped, also skip the update

                if not skip_batch:
                    # Only do gradient operations if batch wasn't skipped
                    # Gradient clipping for stability (increased from 1.0 to 5.0 for deeper networks)
                    if not use_lbfgs:
                        scaler.unscale_(opt)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

                    if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                        ddp_print(f"WARNING: NaN/inf gradients at step {global_step}, skipping update")
                        opt.zero_grad(set_to_none=True)
                        if not use_lbfgs:
                            scaler.update()
                        skip_update = True

                if not skip_update:
                    if use_lbfgs:
                        # LBFGS requires a closure that computes loss and gradients
                        def lbfgs_closure():
                            opt.zero_grad()
                            grid_pred_c, scalar_pred_c = model(x_grid, params)
                            se_p_c = (grid_pred_c[:, 0] - y_grid[:, 0]) ** 2
                            se_T_c = (grid_pred_c[:, 1] - y_grid[:, 1]) ** 2
                            loss_p_c = (se_p_c * valid_mask).sum() / valid_mask.sum().clamp(min=1)
                            loss_T_c = (se_T_c * valid_mask).sum() / valid_mask.sum().clamp(min=1)
                            if grid_out_channels == 3:
                                se_wept_c = (grid_pred_c[:, 2] - y_grid[:, 2]) ** 2
                                loss_wept_c = (se_wept_c * valid_mask).sum() / valid_mask.sum().clamp(min=1)
                            else:
                                loss_wept_c = torch.tensor(0.0, device=device)
                            loss_c = loss_p_c + loss_T_c + args.wept_loss_weight * loss_wept_c
                            loss_c.backward()
                            return loss_c
                        opt.step(lbfgs_closure)
                        # No scheduler for LBFGS
                    else:
                        scaler.step(opt)
                        scaler.update()
                        opt.zero_grad(set_to_none=True)
                        if sched is not None:
                            sched.step()
                global_step += 1

                if args.raw_h5_mode and hasattr(train_ds, "maybe_trigger_cache_load"):
                    train_ds.maybe_trigger_cache_load(global_step)

                # Logging
                if (global_step % args.log_every == 0):
                    if not dist.is_initialized() or dist.get_rank() == 0:
                        lr = opt.param_groups[0]["lr"]
                        # Time per step since last log (NOT average since training start)
                        current_time = time.time()
                        steps_since_last_log = max(1, global_step - last_log_step)
                        time_per_step = (current_time - last_log_time) / steps_since_last_log
                        last_log_time = current_time
                        last_log_step = global_step

                        # Helper to get float value from either tensor or float
                        def to_float(x):
                            return x.item() if hasattr(x, 'item') else float(x)

                        # Comprehensive stdout logging
                        log_msg = f"step {global_step:07d} | loss {to_float(loss):.6e} | lr {lr:.3e} | time/step {time_per_step:.3f}s"
                        log_msg += f"\n  loss_P={to_float(loss_grid_p):.6e} loss_T={to_float(loss_grid_T):.6e}"
                        if grid_out_channels == 3:
                            log_msg += f" loss_WEPT={to_float(loss_grid_wept):.6e}"
                        if not args.predict_grid_only and grid_out_channels == 2:
                            log_msg += f" loss_scalar={to_float(loss_scalar):.6e}"
                        ddp_print(log_msg)

                        if run:
                            log_dict = {
                                "train/loss_total": to_float(loss),
                                "train/loss_grid_p": to_float(loss_grid_p),
                                "train/loss_grid_T": to_float(loss_grid_T),
                                "lr": lr,
                                "step": global_step,
                                "time_per_step": time_per_step,
                            }
                            # Log WEPT loss if in v2.5
                            if grid_out_channels == 3:
                                log_dict["train/loss_grid_wept"] = to_float(loss_grid_wept)
                            # Only log scalar loss if in v2.4 and not grid-only mode
                            if not args.predict_grid_only and grid_out_channels == 2:
                                log_dict["train/loss_scalar"] = to_float(loss_scalar)
                            run.log(log_dict)

                # Periodic memory cleanup to prevent fragmentation
                if (global_step % 100 == 0):
                    torch.cuda.empty_cache()

                # Evaluation
                if (global_step % args.eval_every == 0):
                    # Barrier BEFORE eval to ensure all ranks are synchronized
                    # This prevents NCCL timeout when rank 0 does long rollout eval
                    if dist.is_initialized():
                        dist.barrier()

                    # Clear fragmented memory before evaluation
                    torch.cuda.empty_cache()
                    eval_start_time = time.time()

                    if use_ddp and isinstance(test_loader.sampler, DistributedSampler):
                        test_loader.sampler.set_epoch(global_step)

                    # 1-step eval (teacher forcing)
                    ddp_print(f"\n[EVAL step {global_step}] Starting 1-step evaluation...")
                    onestep_start = time.time()
                    metrics = evaluate(model, test_loader, device, stats, stats["log1p_flags"]["scalar"], max_batches=50, predict_grid_only=args.predict_grid_only, grid_out_channels=grid_out_channels,
                                       acc_abs_p=args.acc_abs_p, acc_abs_t=args.acc_abs_t, acc_abs_wept=args.acc_abs_wept)
                    onestep_time = time.time() - onestep_start

                    # Full rollout eval (NO teacher forcing) - only on rank 0 to avoid redundant computation
                    # Can be skipped with --skip_rollout_eval to avoid NCCL timeout issues
                    rollout_metrics = {}
                    if (not dist.is_initialized() or dist.get_rank() == 0) and args.raw_h5_mode and not args.skip_rollout_eval:
                        ddp_print(f"  Starting full rollout evaluation on test files (timeout={args.rollout_timeout}s)...")
                        rollout_start = time.time()
                        fixed_pad = None
                        if args.fixed_pad_size:
                            fixed_pad = tuple(int(x) for x in args.fixed_pad_size.split(","))
                        try:
                            # Set timeout using signal alarm
                            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
                            signal.alarm(args.rollout_timeout)
                            rollout_metrics = evaluate_rollout(
                                model, args.raw_h5_dir, stats, device,
                                fixed_pad_size=fixed_pad, max_steps=29,
                                acc_abs_p=args.acc_abs_p, acc_abs_t=args.acc_abs_t, acc_abs_wept=args.acc_abs_wept
                            )
                            signal.alarm(0)  # Cancel the alarm
                            signal.signal(signal.SIGALRM, old_handler)  # Restore old handler
                            rollout_time = time.time() - rollout_start
                            ddp_print(f"  Rollout eval completed in {rollout_time:.1f}s")
                        except TimeoutError:
                            signal.alarm(0)  # Cancel the alarm
                            ddp_print(f"  WARNING: Rollout eval timed out after {args.rollout_timeout}s - skipping this eval")
                            rollout_metrics = {}  # Empty metrics on timeout
                        except Exception as e:
                            signal.alarm(0)  # Cancel the alarm
                            ddp_print(f"  WARNING: Rollout eval failed with error: {e} - skipping this eval")
                            rollout_metrics = {}  # Empty metrics on error

                    # Compute validation score
                    if grid_out_channels == 3:
                        # v2.5 mode: sum P, T, WEPT losses
                        val_score = metrics["mse/grid_pressure"] + metrics["mse/grid_temperature"] + metrics["mse/grid_wept"]
                    elif args.predict_grid_only:
                        # v2.4 grid-only mode: sum P, T losses
                        val_score = metrics["mse/grid_pressure"] + metrics["mse/grid_temperature"]
                    else:
                        # v2.4 full mode: grid + scalars
                        val_score = metrics["mse/grid_pressure"] + metrics["mse/grid_temperature"] + sum(
                            metrics[f"mse/{name}"] for name in stats["scalar_channels"]
                        ) / len(stats["scalar_channels"])

                    eval_total_time = time.time() - eval_start_time

                    if not dist.is_initialized() or dist.get_rank() == 0:
                        # Comprehensive eval logging to stdout
                        eval_msg = f"[EVAL step {global_step}] COMPLETED in {eval_total_time:.1f}s (1-step: {onestep_time:.1f}s)"
                        eval_msg += f"\n  val_score (MSE_P+MSE_T+MSE_WEPT for best ckpt selection)={val_score:.6e}"
                        eval_msg += f"\n  [1-STEP] MSE - P:{metrics['mse/grid_pressure']:.6e} T:{metrics['mse/grid_temperature']:.6e}"
                        if grid_out_channels == 3:
                            eval_msg += f" WEPT:{metrics['mse/grid_wept']:.6e}"
                        eval_msg += f"\n  [1-STEP] ACC_ABS (P<{args.acc_abs_p}bar, T<{args.acc_abs_t}C) - P:{metrics['acc5/grid_pressure']:.4f} T:{metrics['acc5/grid_temperature']:.4f}"
                        if grid_out_channels == 3:
                            eval_msg += f" WEPT:{metrics['acc5/grid_wept']:.4f}"

                        # Log rollout metrics if available
                        if rollout_metrics:
                            eval_msg += f"\n  [ROLLOUT 29-step] MSE - P:{rollout_metrics['rollout/mse_p']:.6e} T:{rollout_metrics['rollout/mse_t']:.6e} WEPT:{rollout_metrics['rollout/mse_w']:.6e}"
                            eval_msg += f"\n  [ROLLOUT 29-step] ACC_ABS - P:{rollout_metrics['rollout/acc5_p']:.4f} T:{rollout_metrics['rollout/acc5_t']:.4f} WEPT:{rollout_metrics['rollout/acc5_w']:.4f}"
                            eval_msg += f" (files: {rollout_metrics.get('rollout/files_evaluated', 0)})"

                        ddp_print(eval_msg)

                        # Update scheduled sampling based on acc5 performance
                        if args.scheduled_sampling:
                            # Compute average acc5 across P, T, (WEPT if v2.5)
                            acc5_vals = [metrics["acc5/grid_pressure"], metrics["acc5/grid_temperature"]]
                            if grid_out_channels == 3:
                                acc5_vals.append(metrics["acc5/grid_wept"])
                            avg_acc5 = sum(acc5_vals) / len(acc5_vals)

                            # Add to history
                            acc5_history.append((global_step, avg_acc5))
                            # Keep only recent window
                            if len(acc5_history) > args.sched_samp_window:
                                acc5_history.pop(0)

                            # Compute rolling average
                            rolling_avg_acc5 = sum(a for _, a in acc5_history) / max(1, len(acc5_history))

                            # Compute scheduled sampling probability based on rolling average
                            if rolling_avg_acc5 < args.sched_samp_start_acc5:
                                current_sched_samp_prob = 0.0
                            elif rolling_avg_acc5 >= args.sched_samp_end_acc5:
                                current_sched_samp_prob = args.sched_samp_final_prob
                            else:
                                # Linear ramp between start and end thresholds
                                # Guard against division by zero and clamp progress to [0, 1]
                                denom = args.sched_samp_end_acc5 - args.sched_samp_start_acc5
                                if denom <= 0:
                                    progress = 1.0  # If bad config, use full prob
                                else:
                                    progress = (rolling_avg_acc5 - args.sched_samp_start_acc5) / denom
                                progress = max(0.0, min(1.0, progress))  # Clamp to [0, 1]
                                current_sched_samp_prob = progress * args.sched_samp_final_prob

                            # Final safety clamp on probability
                            current_sched_samp_prob = max(0.0, min(1.0, current_sched_samp_prob))

                            ddp_print(f"  [sched_samp] rolling_avg_acc5={rolling_avg_acc5*100:.2f}%, prob={current_sched_samp_prob*100:.1f}%")

                        if run:
                            log_dict = {}
                            for k, v in metrics.items():
                                log_dict[f"test/{k}"] = v
                            # Log rollout metrics to wandb
                            if rollout_metrics:
                                for k, v in rollout_metrics.items():
                                    log_dict[f"test/{k}"] = v
                            # Log scheduled sampling info
                            if args.scheduled_sampling:
                                log_dict["scheduled_sampling/probability"] = current_sched_samp_prob
                                log_dict["scheduled_sampling/rolling_avg_acc5"] = rolling_avg_acc5
                            log_dict["step"] = global_step
                            run.log(log_dict)

                        # Save best
                        if val_score < best_val:
                            best_val = val_score
                            ckpt_path = os.path.join(args.save_dir, f"best_step{global_step}.pt")
                            state = {
                                "model": model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
                                "optimizer": opt.state_dict(),
                                "scheduler": sched.state_dict() if sched else None,
                                "scaler": scaler.state_dict(),
                                "step": global_step,
                                "best_val": best_val,
                                "args": vars(args)
                            }
                            torch.save(state, ckpt_path)
                            ddp_print(f"Saved best checkpoint: {ckpt_path}")

                    # CRITICAL FIX: Broadcast scheduled sampling probability from rank 0 to all ranks
                    # Without this, rank 0 would have updated current_sched_samp_prob but other ranks
                    # would still have the old value, causing them to take different code paths
                    # in the training loop (rank 0 enters scheduled_sampling_step, others do normal training)
                    # which leads to NCCL deadlock.
                    if args.scheduled_sampling and dist.is_initialized():
                        prob_tensor = torch.tensor([current_sched_samp_prob], device=device, dtype=torch.float32)
                        dist.broadcast(prob_tensor, src=0)
                        current_sched_samp_prob = prob_tensor.item()

                    # Barrier after eval to sync all ranks (eval can take 180s+ on rank 0)
                    if dist.is_initialized():
                        dist.barrier()

                # Periodic checkpoint
                if (global_step % args.ckpt_every == 0):
                    if not dist.is_initialized() or dist.get_rank() == 0:
                        ckpt_path = os.path.join(args.save_dir, f"step{global_step}.pt")
                        state = {
                            "model": model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
                            "optimizer": opt.state_dict(),
                            "scheduler": sched.state_dict() if sched else None,
                            "scaler": scaler.state_dict(),
                            "step": global_step,
                            "best_val": best_val,
                            "args": vars(args)
                        }
                        torch.save(state, ckpt_path)
                        ddp_print(f"Saved checkpoint: {ckpt_path}")

                # Barrier after checkpoint save to sync all ranks before next iteration
                if dist.is_initialized():
                    dist.barrier()

                if global_step >= args.max_steps:
                    break

        if global_step >= args.max_steps:
            break

    ddp_print("Training complete!")
    if run:
        run.finish()


if __name__ == "__main__":
    main()
