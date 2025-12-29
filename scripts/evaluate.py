"""
Comprehensive evaluation script for v2.5 models.

Performs both:
1. 1-step evaluation (MSE and Acc5 on single-step predictions)
2. 29-step rollout evaluation (autoregressive without teacher forcing)

Outputs per-timestep metrics and generates a figure.

Usage:
    CUDA_VISIBLE_DEVICES=9 python scripts/evaluate.py --checkpoint checkpoints_v25/best_step8100.pt
"""
import os
import sys
import json
import argparse
import time

# Add parent directory to path to import from voxel_ode
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import h5py

from voxel_ode.model import VoxelAutoRegressor, LinearBaseline


def load_model_from_checkpoint(checkpoint_path, stats, device):
    """Load model from checkpoint file."""
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Determine model architecture from stats
    C_static = len(stats["static_channels"])  # 8

    # v2.5: 14 input channels (8 static + 3 from t-1 + 3 from t)
    in_channels = C_static + 3 + 3  # 14
    grid_out_channels = 3  # P, T, WEPT

    state_dict = checkpoint["model"]
    is_baseline = "grid_head.weight" in state_dict and "conv1.weight" not in str(state_dict.keys())

    if is_baseline:
        model = LinearBaseline(
            in_channels=in_channels,
            grid_out_channels=grid_out_channels,
            scalar_out_dim=5,
            cond_params_dim=26,
            use_param_broadcast=False,
        )
        print(f"Created LinearBaseline model")
    else:
        base_channels = 32
        depth = 4
        r = 2

        # Try to infer from state dict
        for key in state_dict.keys():
            if "stem.0.weight" in key:
                # stem.0.weight has shape [base_channels, in_channels, k, k, k]
                # where k = 2*r + 1, so r = (k - 1) / 2
                base_channels = state_dict[key].shape[0]
                k = state_dict[key].shape[2]  # kernel size
                r = (k - 1) // 2
                break

        model = VoxelAutoRegressor(
            in_channels=in_channels,
            base_channels=base_channels,
            depth=depth,
            r=r,
            cond_params_dim=26,
            use_param_broadcast=False,
            grid_out_channels=grid_out_channels,
            scalar_out_dim=5,
            use_checkpoint=False,
        )
        print(f"Created VoxelAutoRegressor model with base_channels={base_channels}, r={r} (kernel {2*r+1}x{2*r+1}x{2*r+1})")

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    step = checkpoint.get("step", 0)
    print(f"Loaded model from step {step}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    return model, step


def evaluate_rollout_per_timestep(model, raw_h5_dir, stats, device, test_files, max_steps=29, save_path=None, baseline=None):
    """
    Perform full autoregressive rollout evaluation with per-timestep metrics.

    This is TRUE autoregressive evaluation - NO teacher forcing.
    Model predictions are fed back as inputs.

    Matches train_ddp.py evaluate_rollout aggregation:
    - Average across timesteps per file first
    - Then average across files for summary

    Args:
        save_path: If provided, saves predictions and ground truth to H5 file

    Returns:
        per_timestep: dict with lists of per-timestep MSE and Acc5 for P, T, WEPT
        summary: dict with averaged metrics (matches train_ddp.py)
    """
    if model is not None:
        model.eval()

    # Load stats for normalization
    static_mean = torch.tensor(stats["mean"]["static"], device=device).view(-1, 1, 1, 1)
    static_std = torch.tensor(stats["std"]["static"], device=device).view(-1, 1, 1, 1).clamp(min=1e-6)
    grid_input_mean = torch.tensor(stats["mean"]["grid_input_absolute"], device=device)
    grid_input_std = torch.tensor(stats["std"]["grid_input_absolute"], device=device).clamp(min=1e-6)
    grid_output_std = torch.tensor(stats["std"]["grid_output_delta"], device=device).clamp(min=1e-6)

    STATIC_KEYS = ["FaultId", "InjRate", "IsActive", "IsWell", "PermX", "PermY", "PermZ", "Porosity"]

    # Per-timestep accumulators (list of lists, one per file)
    timestep_mse_p = [[] for _ in range(max_steps)]
    timestep_mse_t = [[] for _ in range(max_steps)]
    timestep_mse_w = [[] for _ in range(max_steps)]
    timestep_acc5_p = [[] for _ in range(max_steps)]
    timestep_acc5_t = [[] for _ in range(max_steps)]
    timestep_acc5_w = [[] for _ in range(max_steps)]
    actual_max_step = 0  # Track the actual max timestep with data

    # Per-file accumulators for summary (matches train_ddp.py)
    all_mse_p, all_mse_t, all_mse_w = [], [], []
    all_acc5_p, all_acc5_t, all_acc5_w = [], [], []

    # For saving predictions (only used if save_path is provided)
    all_predictions = {}

    base_model = model.module if (model is not None and hasattr(model, 'module')) else model
    files_processed = 0
    model_time = 0.0  # Accumulate only forward pass time (not file I/O)

    with torch.no_grad():
        for fname in test_files:
            fpath = os.path.join(raw_h5_dir, fname)
            if not os.path.exists(fpath):
                continue

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

            # Normalize static
            static_t = torch.from_numpy(static).float().to(device)
            static_norm = (static_t - static_mean) / static_std
            static_norm[static_t == -999] = 0
            static_norm = static_norm.unsqueeze(0)

            params_t = torch.from_numpy(params).float().to(device).unsqueeze(0)

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

            # Arrays to store predictions if save_path provided
            if save_path:
                pred_p = np.full((T, Z, Y, X), -999.0, dtype=np.float32)
                pred_t = np.full((T, Z, Y, X), -999.0, dtype=np.float32)
                pred_w = np.full((T, Z, Y, X), -999.0, dtype=np.float32)
                pred_p[0] = p_true[0]
                pred_t[0] = t_true[0]
                pred_w[0] = w_true[0]

            # Full AR rollout - NO teacher forcing
            # prev = t-1 state, curr = t state
            prev_p, prev_t, prev_w = torch.zeros_like(p_ar), torch.zeros_like(t_ar), torch.zeros_like(w_ar)
            curr_p, curr_t, curr_w = p_ar.clone(), t_ar.clone(), w_ar.clone()

            for step in range(rollout_steps):
                # Build input [static(8), P_{t-1}, T_{t-1}, W_{t-1}, P_t, T_t, W_t]
                x_input = torch.zeros(1, 14, Z, Y, X, device=device, dtype=torch.float32)
                x_input[0, :8] = static_norm[0]

                # Normalize inputs using ABSOLUTE stats (for P, T, WEPT values)
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

                # Forward pass - model outputs NORMALIZED RESIDUALS
                t0 = time.time()
                if baseline == "copy":
                    # Copy baseline: predict delta=0 (propagate current values)
                    grid_pred_norm = torch.zeros(1, 3, Z, Y, X, device=device)
                else:
                    with torch.cuda.amp.autocast(enabled=True):
                        grid_pred_norm, _ = base_model(x_input, params_t)
                    torch.cuda.synchronize()  # Ensure GPU computation is complete
                model_time += time.time() - t0

                # Guard against NaN/inf
                grid_pred_norm = torch.nan_to_num(grid_pred_norm, nan=0.0, posinf=10.0, neginf=-10.0)
                grid_pred_norm = grid_pred_norm.clamp(-10.0, 10.0)

                # Convert normalized residuals to raw deltas using DELTA stats
                delta_p = grid_pred_norm[0, 0] * grid_output_std[0]
                delta_t = grid_pred_norm[0, 1] * grid_output_std[1]
                delta_w = grid_pred_norm[0, 2] * grid_output_std[2]

                # Update AR state: add predicted delta to current state
                prev_p, prev_t, prev_w = curr_p.clone(), curr_t.clone(), curr_w.clone()
                curr_p = torch.where(mask_p, curr_p + delta_p, curr_p)
                curr_t = torch.where(mask_t, curr_t + delta_t, curr_t)
                curr_w = torch.where(mask_w, curr_w + delta_w, curr_w)

                # Clamp to physical bounds (match train_ddp.py exactly)
                curr_p = torch.nan_to_num(curr_p, nan=0.0, posinf=1000.0, neginf=0.0)
                curr_t = torch.nan_to_num(curr_t, nan=0.0, posinf=500.0, neginf=-50.0)
                curr_w = torch.nan_to_num(curr_w, nan=0.0, posinf=1e15, neginf=0.0)
                curr_p = curr_p.clamp(0.0, 1000.0)
                curr_t = curr_t.clamp(-50.0, 500.0)
                curr_w = curr_w.clamp(0.0, 1e13)

                # Store predictions if save_path provided
                target_t = step + 1
                if save_path and target_t < T:
                    pred_p[target_t] = curr_p.cpu().numpy()
                    pred_t[target_t] = curr_t.cpu().numpy()
                    pred_w[target_t] = curr_w.cpu().numpy()

                # Compute metrics at this timestep vs ground truth
                if target_t <= max_valid_t:
                    gt_p = torch.from_numpy(p_true[target_t]).float().to(device)
                    gt_t = torch.from_numpy(t_true[target_t]).float().to(device)
                    gt_w = torch.from_numpy(w_true[target_t]).float().to(device)

                    # MSE on ABSOLUTE values
                    def safe_mse(pred, true, mask):
                        if mask.sum() == 0:
                            return None
                        diff = pred[mask] - true[mask]
                        return (diff * diff).mean().item()

                    mse_p = safe_mse(curr_p, gt_p, mask_p)
                    mse_t = safe_mse(curr_t, gt_t, mask_t)
                    mse_w = safe_mse(curr_w, gt_w, mask_w)

                    if mse_p is None or mse_t is None or mse_w is None:
                        continue

                    # ACC_ABS with ABSOLUTE thresholds (not relative)
                    # Pressure: +/- 5 bar, Temperature: +/- 5 C, WEPT: +/- 1e10 J
                    def acc_abs(pred, true, mask, threshold):
                        pm, tm = pred[mask], true[mask]
                        if len(pm) == 0:
                            return 0.0
                        abs_err = (pm - tm).abs()
                        return (abs_err <= threshold).float().mean().item()

                    acc5_p = acc_abs(curr_p, gt_p, mask_p, 5.0)      # +/- 5 bar
                    acc5_t = acc_abs(curr_t, gt_t, mask_t, 5.0)      # +/- 5 C
                    acc5_w = acc_abs(curr_w, gt_w, mask_w, 1e10)     # +/- 1e10 J

                    # Store per-timestep metrics (for per-timestep reporting)
                    timestep_mse_p[step].append(mse_p)
                    timestep_mse_t[step].append(mse_t)
                    timestep_mse_w[step].append(mse_w)
                    timestep_acc5_p[step].append(acc5_p * 100.0)  # Store as percentage
                    timestep_acc5_t[step].append(acc5_t * 100.0)
                    timestep_acc5_w[step].append(acc5_w * 100.0)
                    actual_max_step = max(actual_max_step, step + 1)  # Track actual valid steps

                    # Store per-file metrics (for summary - matches train_ddp.py)
                    file_mse_p.append(mse_p)
                    file_mse_t.append(mse_t)
                    file_mse_w.append(mse_w)
                    file_acc5_p.append(acc5_p)
                    file_acc5_t.append(acc5_t)
                    file_acc5_w.append(acc5_w)

            # Average across timesteps for this file (matches train_ddp.py)
            if len(file_mse_p) > 0:
                all_mse_p.append(np.mean(file_mse_p))
                all_mse_t.append(np.mean(file_mse_t))
                all_mse_w.append(np.mean(file_mse_w))
                all_acc5_p.append(np.mean(file_acc5_p))
                all_acc5_t.append(np.mean(file_acc5_t))
                all_acc5_w.append(np.mean(file_acc5_w))
                files_processed += 1

            # Store predictions for this file
            if save_path:
                all_predictions[fname] = {
                    "pressure_ar": pred_p,
                    "temperature_ar": pred_t,
                    "wept_ar": pred_w,
                    "pressure_true": p_true,
                    "temperature_true": t_true,
                    "wept_true": w_true,
                }

    # Average per-timestep metrics across files (for per-timestep plotting)
    # Only include timesteps that actually have data
    per_timestep = {
        "P_MSE": [], "T_MSE": [], "W_MSE": [],
        "P_Acc5": [], "T_Acc5": [], "W_Acc5": [],
    }

    for step in range(actual_max_step):
        per_timestep["P_MSE"].append(np.mean(timestep_mse_p[step]) if timestep_mse_p[step] else 0.0)
        per_timestep["T_MSE"].append(np.mean(timestep_mse_t[step]) if timestep_mse_t[step] else 0.0)
        per_timestep["W_MSE"].append(np.mean(timestep_mse_w[step]) if timestep_mse_w[step] else 0.0)
        per_timestep["P_Acc5"].append(np.mean(timestep_acc5_p[step]) if timestep_acc5_p[step] else 0.0)
        per_timestep["T_Acc5"].append(np.mean(timestep_acc5_t[step]) if timestep_acc5_t[step] else 0.0)
        per_timestep["W_Acc5"].append(np.mean(timestep_acc5_w[step]) if timestep_acc5_w[step] else 0.0)

    # Summary metrics: average across FILES (matches train_ddp.py exactly)
    summary = {
        "mse_p": np.mean(all_mse_p) if all_mse_p else 0.0,
        "mse_t": np.mean(all_mse_t) if all_mse_t else 0.0,
        "mse_w": np.mean(all_mse_w) if all_mse_w else 0.0,
        "acc5_p": np.mean(all_acc5_p) if all_acc5_p else 0.0,
        "acc5_t": np.mean(all_acc5_t) if all_acc5_t else 0.0,
        "acc5_w": np.mean(all_acc5_w) if all_acc5_w else 0.0,
        "files_evaluated": files_processed,
        "model_time": model_time,  # Total forward pass time only (no file I/O)
        "actual_steps": actual_max_step,  # Actual number of valid timesteps
    }

    # Save predictions to H5 if save_path provided
    if save_path and all_predictions:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        with h5py.File(save_path, "w") as f:
            for fname, data in all_predictions.items():
                grp = f.create_group(fname)
                grp.create_dataset("pressure_ar", data=data["pressure_ar"], compression="gzip")
                grp.create_dataset("temperature_ar", data=data["temperature_ar"], compression="gzip")
                grp.create_dataset("wept_ar", data=data["wept_ar"], compression="gzip")
                grp.create_dataset("pressure_true", data=data["pressure_true"], compression="gzip")
                grp.create_dataset("temperature_true", data=data["temperature_true"], compression="gzip")
                grp.create_dataset("wept_true", data=data["wept_true"], compression="gzip")
        print(f"  Saved predictions to: {save_path}")

    return per_timestep, summary


def plot_per_timestep_metrics(per_timestep, save_path, step):
    """Generate and save per-timestep metrics figure."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Use actual data length (excludes timesteps with no data)
    num_steps = len(per_timestep["P_Acc5"])
    t = list(range(1, num_steps + 1))

    P_Acc5 = per_timestep["P_Acc5"]
    T_Acc5 = per_timestep["T_Acc5"]
    W_Acc5 = per_timestep["W_Acc5"]
    P_MSE = per_timestep["P_MSE"]
    T_MSE = per_timestep["T_MSE"]
    W_MSE = per_timestep["W_MSE"]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Left y-axis for Acc5
    ax1.plot(t, P_Acc5, marker='o', label='P_Acc5')
    ax1.plot(t, T_Acc5, marker='s', label='T_Acc5')
    ax1.plot(t, W_Acc5, marker='^', label='W_Acc5')
    ax1.set_xlabel('t')
    ax1.set_ylabel('Accuracy (%)')
    ax1.grid(True)

    # Right y-axis for MSE
    ax2 = ax1.twinx()
    ax2.plot(t, P_MSE, linestyle='--', marker='o', label='P_MSE')
    ax2.plot(t, T_MSE, linestyle='--', marker='s', label='T_MSE')
    ax2.plot(t, W_MSE, linestyle='--', marker='^', label='W_MSE')
    ax2.set_ylabel('MSE')

    # Combine legends
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

    plt.title(f'Per-timestep P/T/W Metrics vs t (step {step})')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved figure to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Comprehensive evaluation for v2.5 models")

    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pt file)")
    parser.add_argument("--raw_h5_dir", type=str, default="/workspace/all_oak_data/h5s_v2.5_data",
                        help="Directory with raw v2.5_*.h5 files")
    parser.add_argument("--stats_path", type=str, default="/workspace/omv_v2.5/data/raw_h5_stats_full.json",
                        help="Path to stats.json for normalization")
    parser.add_argument("--test_files", type=str, nargs="+", default=None,
                        help="List of test files (default: v2.5_0001.h5 to v2.5_0005.h5)")
    parser.add_argument("--max_steps", type=int, default=29,
                        help="Maximum rollout steps (default: 29)")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device to run on")
    parser.add_argument("--output_dir", type=str, default="/workspace/omv_v2.5",
                        help="Directory to save output figure")
    parser.add_argument("--save_path", type=str, default=None,
                        help="Path to save predictions H5 file (optional)")
    parser.add_argument("--skip_1step", action="store_true",
                        help="Skip 1-step evaluation")
    parser.add_argument("--skip_rollout", action="store_true",
                        help="Skip rollout evaluation")
    parser.add_argument("--baseline", type=str, choices=["copy", "linear"], default=None,
                        help="Run baseline instead of model: 'copy' (delta=0) or 'linear' (untrained linear)")

    args = parser.parse_args()

    if args.baseline is None and not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not os.path.exists(args.stats_path):
        raise FileNotFoundError(f"Stats file not found: {args.stats_path}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading stats from {args.stats_path}...")
    with open(args.stats_path, "r") as f:
        stats = json.load(f)

    # Handle baseline modes
    if args.baseline == "copy":
        model = None  # No model needed for copy baseline
        step = "copy"
        print("Running COPY baseline (delta=0, propagate current values)")
    elif args.baseline == "linear":
        # Create untrained linear baseline
        C_static = len(stats["static_channels"])
        in_channels = C_static + 3 + 3  # 14
        model = LinearBaseline(in_channels=in_channels, grid_out_channels=3, scalar_out_dim=5)
        model = model.to(device)
        model.eval()
        step = "linear"
        print(f"Running LINEAR baseline (untrained, {sum(p.numel() for p in model.parameters()):,} params)")
    else:
        model, step = load_model_from_checkpoint(args.checkpoint, stats, device)

    if args.test_files is None:
        test_files = [f"v2.5_{i:04d}.h5" for i in range(1, 6)]
    else:
        test_files = args.test_files

    print(f"\nEvaluating on {len(test_files)} files:")
    for f in test_files:
        print(f"  - {f}")

    # ========== ROLLOUT EVALUATION ==========
    if not args.skip_rollout:
        print(f"\nStarting {args.max_steps}-step rollout evaluation (NO teacher forcing)...")
        start_rollout = time.time()

        per_timestep, summary = evaluate_rollout_per_timestep(
            model=model,
            raw_h5_dir=args.raw_h5_dir,
            stats=stats,
            device=device,
            test_files=test_files,
            max_steps=args.max_steps,
            save_path=args.save_path,
            baseline=args.baseline,
        )

        elapsed_rollout = time.time() - start_rollout

        # Print results
        print(f"\n{'='*70}")
        print(f"ROLLOUT EVALUATION RESULTS (step {step})")
        print(f"{'='*70}")
        model_time = summary['model_time']
        print(f"  Model forward pass time: {model_time:.2f}s ({model_time/len(test_files):.2f}s per file)")
        print(f"  Total wallclock time: {elapsed_rollout:.2f}s (includes file I/O)")
        print(f"  Files evaluated: {summary['files_evaluated']}")
        actual_steps = summary['actual_steps']
        print(f"\n  [ROLLOUT {actual_steps}-step] MSE - P:{summary['mse_p']:.6e} T:{summary['mse_t']:.6e} WEPT:{summary['mse_w']:.6e}")
        print(f"  [ROLLOUT {actual_steps}-step] ACC_ABS - P:{summary['acc5_p']:.4f} T:{summary['acc5_t']:.4f} WEPT:{summary['acc5_w']:.4f}")
        print(f"  (ACC_ABS thresholds: P +/-5 bar, T +/-5 C, WEPT +/-1e10 J)")

        # Print per-timestep table
        print(f"\n{'='*70}")
        print("PER-TIMESTEP METRICS")
        print(f"{'='*70}")
        print(f"{'t':>3}  {'P_Acc5':>8}  {'P_MSE':>12}  {'T_Acc5':>8}  {'T_MSE':>12}  {'W_Acc5':>8}  {'W_MSE':>12}")
        print("-" * 70)
        for i in range(len(per_timestep["P_MSE"])):
            print(f"{i+1:3d}  {per_timestep['P_Acc5'][i]:7.2f}%  {per_timestep['P_MSE'][i]:12.6f}  "
                  f"{per_timestep['T_Acc5'][i]:7.2f}%  {per_timestep['T_MSE'][i]:12.6f}  "
                  f"{per_timestep['W_Acc5'][i]:7.2f}%  {per_timestep['W_MSE'][i]:12.6e}")
        print(f"{'='*70}")

        # Save figure
        fig_path = os.path.join(args.output_dir, f"plots/rollout_metrics_step{step}.png")
        os.makedirs(os.path.dirname(fig_path), exist_ok=True)
        plot_per_timestep_metrics(per_timestep, fig_path, step)

    print("\nDone!")


if __name__ == "__main__":
    main()
