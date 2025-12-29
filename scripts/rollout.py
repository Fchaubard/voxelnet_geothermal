"""
Standalone rollout evaluation script for v2.5 models.

Usage:
    python scripts/rollout.py --checkpoint /path/to/checkpoint.pt --test_files v2.5_0001.h5

This script imports the evaluate_rollout function from train_ddp.py to ensure
identical behavior between training evaluation and standalone inference.
"""
import os
import sys
import json
import argparse
import time

# Add parent directory to path to import from voxel_ode
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from voxel_ode.model import VoxelAutoRegressor, LinearBaseline

# Import evaluate_rollout from train_ddp
from scripts.train_ddp import evaluate_rollout


def load_model_from_checkpoint(checkpoint_path, stats, device):
    """
    Load model from checkpoint file.

    Args:
        checkpoint_path: Path to .pt checkpoint file
        stats: Statistics dict for determining model architecture
        device: Device to load model on

    Returns:
        model: Loaded model in eval mode
        step: Training step from checkpoint
    """
    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Determine model architecture from stats
    C_static = len(stats["static_channels"])  # 8
    data_format = stats.get("data_format", "v2.5")

    # v2.5: 14 input channels (8 static + 3 from t-1 + 3 from t)
    in_channels = C_static + 3 + 3  # 14
    grid_out_channels = 3  # P, T, WEPT

    # Check if model state dict has baseline keys (simple linear model)
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
        # Infer model hyperparameters from state dict
        # base_channels can be inferred from conv1.weight shape
        base_channels = 32  # Default
        depth = 4  # Default
        r = 2  # Default

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

    # Load weights
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    # Get step number
    step = checkpoint.get("step", 0)
    print(f"Loaded model from step {step}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    return model, step


def main():
    parser = argparse.ArgumentParser(description="Standalone rollout evaluation for v2.5 models")

    # Required arguments
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pt file)")

    # Data arguments
    parser.add_argument("--raw_h5_dir", type=str, default="/workspace/all_oak_data/h5s_v2.5_data",
                        help="Directory with raw v2.5_*.h5 files")
    parser.add_argument("--stats_path", type=str, default="/workspace/omv_v2.5/stats.json",
                        help="Path to stats.json for normalization")
    parser.add_argument("--test_files", type=str, nargs="+", default=None,
                        help="List of test files (default: v2.5_0001.h5 to v2.5_0005.h5)")

    # Evaluation arguments
    parser.add_argument("--max_steps", type=int, default=29,
                        help="Maximum rollout steps (default: 29 for 30-step prediction)")
    parser.add_argument("--device", type=str, default="cuda:9",
                        help="Device to run on (default: cuda:9 - reserved for inference)")

    args = parser.parse_args()

    # Validate paths
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not os.path.exists(args.stats_path):
        raise FileNotFoundError(f"Stats file not found: {args.stats_path}")
    if not os.path.exists(args.raw_h5_dir):
        raise FileNotFoundError(f"Raw H5 directory not found: {args.raw_h5_dir}")

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load stats
    print(f"Loading stats from {args.stats_path}...")
    with open(args.stats_path, "r") as f:
        stats = json.load(f)

    # Load model
    model, step = load_model_from_checkpoint(args.checkpoint, stats, device)

    # Set test files
    if args.test_files is None:
        test_files = [f"v2.5_{i:04d}.h5" for i in range(1, 6)]
    else:
        test_files = args.test_files

    print(f"\nRunning rollout evaluation on {len(test_files)} files:")
    for f in test_files:
        print(f"  - {f}")

    # Run rollout evaluation
    print(f"\nStarting {args.max_steps}-step rollout evaluation...")
    start_time = time.time()

    metrics = evaluate_rollout(
        model=model,
        raw_h5_dir=args.raw_h5_dir,
        stats=stats,
        device=device,
        fixed_pad_size=None,  # Use native resolution
        test_files=test_files,
        max_steps=args.max_steps,
    )

    elapsed = time.time() - start_time
    print(f"Rollout evaluation completed in {elapsed:.1f}s")

    # Print results in the same format as train_ddp.py
    print(f"\n{'='*60}")
    print(f"ROLLOUT EVALUATION RESULTS (step {step})")
    print(f"{'='*60}")

    mse_p = metrics["rollout/mse_p"]
    mse_t = metrics["rollout/mse_t"]
    mse_w = metrics["rollout/mse_w"]
    acc5_p = metrics["rollout/acc5_p"]
    acc5_t = metrics["rollout/acc5_t"]
    acc5_w = metrics["rollout/acc5_w"]
    files_eval = metrics.get("rollout/files_evaluated", len(test_files))

    print(f"  [ROLLOUT {args.max_steps}-step] MSE - P:{mse_p:.6e} T:{mse_t:.6e} WEPT:{mse_w:.6e}")
    print(f"  [ROLLOUT {args.max_steps}-step] ACC5 - P:{acc5_p:.4f} T:{acc5_t:.4f} WEPT:{acc5_w:.4f} (files: {files_eval})")
    print(f"{'='*60}")

    # Also print per-file metrics if evaluating single file
    if len(test_files) == 1:
        print(f"\nSingle file evaluation: {test_files[0]}")

    return metrics


if __name__ == "__main__":
    main()
