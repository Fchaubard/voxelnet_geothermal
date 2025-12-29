"""Data preparation: pack HDF5 files, compute statistics, create patch indices."""
import os
import json
import math
import argparse
import h5py
import numpy as np
from tqdm import tqdm

from .utils import parse_run_index_from_name, Welford

STATIC_KEYS = [
    "FaultId", "InjRate", "IsActive", "IsWell",
    "PermX", "PermY", "PermZ", "Porosity",
]
GRID_OUTPUT_KEYS_V24 = ["Pressure", "Temperature"]
GRID_OUTPUT_KEYS_V25 = ["Pressure", "Temperature", "WEPT"]
SCALAR_OUTPUT_KEYS = [
    "FieldEnergyInjectionRate",
    "FieldEnergyProductionRate",
    "FieldEnergyProductionTotal",
    "FieldWaterInjectionRate",
    "FieldWaterProductionRate",
]


def pack_one_file(src_path, dst_path, dtype="float32", data_format="v2.4"):
    """
    Pack a v2.4 or v2.5 HDF5 file into a faster prepped format.

    Input structure (v2.4 and v2.5):
        Input/FaultId, InjRate, IsActive, IsWell, PermX, PermY, PermZ,
              Porosity, Pressure0, Temperature0, ParamsScalar
        Output/Pressure, Temperature, Field*
        Output/WEPT (v2.5 only - per-voxel energy production)

    Output structure (prepped):
        /static: [C_static, Z, Y, X]
        /params_scalar: [26]
        /outputs_grid: [T, 2 or 3, Z, Y, X]  (v2.4: P,T / v2.5: P,T,WEPT)
        /outputs_scalar: [T, 5]  (Energy/Water fields)
    """
    with h5py.File(src_path, "r") as f:
        Z, Y, X = f["Input/Pressure0"].shape
        T = f["Output/Pressure"].shape[0]

        # Stack static inputs
        static_list = []
        for k in STATIC_KEYS:
            arr = f[f"Input/{k}"][...]
            static_list.append(arr)
        static = np.stack(static_list, axis=0).astype(dtype, copy=False)

        params = f["Input/ParamsScalar"][...].astype(dtype, copy=False)

        # Stack grid outputs (depends on data format)
        pressure = f["Output/Pressure"][...].astype(dtype, copy=False)  # (T, Z, Y, X)
        temperature = f["Output/Temperature"][...].astype(dtype, copy=False)

        if data_format == "v2.5":
            # v2.5 includes WEPT as a 3D grid field
            if "Output/WEPT" in f:
                wept = f["Output/WEPT"][...].astype(dtype, copy=False)
                outputs_grid = np.stack([pressure, temperature, wept], axis=1)  # (T, 3, Z, Y, X)
            else:
                raise KeyError(f"v2.5 format specified but Output/WEPT not found in {src_path}")
        else:
            # v2.4 only has Pressure and Temperature
            outputs_grid = np.stack([pressure, temperature], axis=1)  # (T, 2, Z, Y, X)

        # Stack scalar outputs
        scalars = np.stack([
            f["Output/FieldEnergyInjectionRate"][...],
            f["Output/FieldEnergyProductionRate"][...],
            f["Output/FieldEnergyProductionTotal"][...],
            f["Output/FieldWaterInjectionRate"][...],
            f["Output/FieldWaterProductionRate"][...],
        ], axis=-1).astype(dtype, copy=False)  # (T, 5)

    # Write packed file
    with h5py.File(dst_path, "w") as g:
        g.create_dataset("static", data=static, compression="gzip", compression_opts=4)
        g.create_dataset("params_scalar", data=params, compression="gzip", compression_opts=4)
        g.create_dataset("outputs_grid", data=outputs_grid, compression="gzip", compression_opts=4)
        g.create_dataset("outputs_scalar", data=scalars, compression="gzip", compression_opts=4)
        # Store data format version for reference
        g.attrs["data_format"] = data_format

    return (static.shape, params.shape, outputs_grid.shape, scalars.shape)


def compute_stats(train_paths, log_transform_set, data_format="v2.4"):
    """
    Compute per-channel mean/std using streaming Welford algorithm.

    For grid outputs, computes statistics on TEMPORAL DELTAS (residuals):
        delta_P = P(t+1) - P(t)
        delta_T = T(t+1) - T(t)
        delta_WEPT = WEPT(t+1) - WEPT(t)

    This is critical for residual prediction training.

    Args:
        train_paths: List of prepped HDF5 paths
        log_transform_set: Set of scalar channel indices to apply log1p before stats
        data_format: "v2.4" or "v2.5"

    Returns:
        Dictionary with mean, std, and log1p flags for each channel type
    """
    # Determine grid channels based on format
    grid_channels = GRID_OUTPUT_KEYS_V25 if data_format == "v2.5" else GRID_OUTPUT_KEYS_V24
    num_grid_channels = len(grid_channels)

    stats = {
        # EXPLICIT KEY NAMING:
        # - Static inputs normalized by ABSOLUTE values
        # - Grid INPUTS normalized by ABSOLUTE values (current state)
        # - Grid OUTPUTS normalized by TEMPORAL DELTA values (predicted residuals)
        # - Scalar outputs normalized by ABSOLUTE values (with optional log1p)
        "mean": {
            "static_input_absolute": [],      # Static channel means (absolute values)
            "grid_input_absolute": [],         # Grid INPUT means (absolute values for current state)
            "grid_output_delta": [],           # Grid OUTPUT means (temporal deltas for targets)
            "scalar_output_absolute": [],      # Scalar channel means (absolute or log1p)
            # Keep old keys for backward compatibility (use delta stats for outputs)
            "static": [],
            "grid": [],
            "scalar": []
        },
        "std": {
            "static_input_absolute": [],       # Static channel stds (absolute values)
            "grid_input_absolute": [],          # Grid INPUT stds (absolute values for current state)
            "grid_output_delta": [],            # Grid OUTPUT stds (temporal deltas for targets)
            "scalar_output_absolute": [],       # Scalar channel stds (absolute or log1p)
            # Keep old keys for backward compatibility (use delta stats for outputs)
            "static": [],
            "grid": [],
            "scalar": []
        },
        "log1p_flags": {"grid": [False] * num_grid_channels, "scalar": []},
        "static_channels": STATIC_KEYS,
        "grid_channels": grid_channels,
        "scalar_channels": SCALAR_OUTPUT_KEYS,
        "data_format": data_format,
        "grid_input_stats_computed_on": "absolute_values",    # For normalizing input grids
        "grid_output_stats_computed_on": "temporal_deltas",   # For normalizing output targets
        "static_stats_computed_on": "absolute_values",
        "scalar_stats_computed_on": "absolute_values_or_log1p",
    }

    # Initialize Welford accumulators - need TWO sets for grid channels
    static_w = [Welford() for _ in range(len(STATIC_KEYS))]
    grid_input_w = [Welford() for _ in range(num_grid_channels)]   # For absolute values
    grid_output_w = [Welford() for _ in range(num_grid_channels)]  # For deltas
    scalar_w = [Welford() for _ in range(5)]

    for p in tqdm(train_paths, desc="Computing stats"):
        with h5py.File(p, "r") as f:
            # Static channels
            s = f["static"]
            for ci in range(s.shape[0]):
                static_w[ci].update(s[ci, ...])

            # Grid channels - compute BOTH absolute (for inputs) and delta (for outputs) stats
            g = f["outputs_grid"]  # (T, 2 or 3, Z, Y, X)
            T = g.shape[0]

            for ch in range(min(num_grid_channels, g.shape[1])):
                # 1. Compute ABSOLUTE VALUE stats (for normalizing inputs)
                for t in range(T):
                    vals = g[t, ch, ...]
                    valid_mask = (vals != -999)
                    if valid_mask.any():
                        grid_input_w[ch].update(vals[valid_mask])

                # 2. Compute TEMPORAL DELTA stats (for normalizing output targets)
                for t in range(T - 1):  # Note: T-1 since we compute deltas
                    curr = g[t, ch, ...]
                    next_val = g[t + 1, ch, ...]
                    delta = next_val - curr
                    # Only include non-sentinel deltas in stats (both t and t+1 must be valid)
                    valid_mask = (curr != -999) & (next_val != -999)
                    if valid_mask.any():
                        grid_output_w[ch].update(delta[valid_mask])

            # Scalar outputs (with optional log1p)
            sc = f["outputs_scalar"]  # (T, 5)
            for ch in range(5):
                vals = sc[:, ch]
                if ch in log_transform_set:
                    vals = np.log1p(np.clip(vals, a_min=0, a_max=None))
                scalar_w[ch].update(vals)

    # Finalize statistics - populate both new verbose keys and old keys for compatibility
    for w in static_w:
        m, v = w.finalize()
        s = math.sqrt(v) if v > 0 else 1.0
        stats["mean"]["static_input_absolute"].append(m)
        stats["std"]["static_input_absolute"].append(s)
        stats["mean"]["static"].append(m)  # Backward compatibility
        stats["std"]["static"].append(s)    # Backward compatibility

    # Grid channels: populate BOTH absolute (input) and delta (output) stats
    for w_input, w_output in zip(grid_input_w, grid_output_w):
        # Absolute value stats (for inputs)
        m_in, v_in = w_input.finalize()
        s_in = math.sqrt(v_in) if v_in > 0 else 1.0
        stats["mean"]["grid_input_absolute"].append(m_in)
        stats["std"]["grid_input_absolute"].append(s_in)

        # Delta stats (for outputs)
        m_out, v_out = w_output.finalize()
        s_out = math.sqrt(v_out) if v_out > 0 else 1.0
        stats["mean"]["grid_output_delta"].append(m_out)
        stats["std"]["grid_output_delta"].append(s_out)

        # Backward compatibility: use delta stats for old "grid" key (since outputs use deltas)
        stats["mean"]["grid"].append(m_out)
        stats["std"]["grid"].append(s_out)

    for idx, w in enumerate(scalar_w):
        m, v = w.finalize()
        s = math.sqrt(v) if v > 0 else 1.0
        stats["mean"]["scalar_output_absolute"].append(m)
        stats["std"]["scalar_output_absolute"].append(s)
        stats["mean"]["scalar"].append(m)  # Backward compatibility
        stats["std"]["scalar"].append(s)    # Backward compatibility
        stats["log1p_flags"]["scalar"].append(idx in log_transform_set)

    return stats


def tile_indices(shape_zyx, patch, stride):
    """
    Generate sliding window patch coordinates.

    Returns list of (z0, z1, y0, y1, x0, x1) tuples.
    """
    Z, Y, X = shape_zyx
    pz, py, px = patch
    sz, sy, sx = stride

    zs = list(range(0, max(1, Z - pz + 1), sz)) or [0]
    ys = list(range(0, max(1, Y - py + 1), sy)) or [0]
    xs = list(range(0, max(1, X - px + 1), sx)) or [0]

    # Ensure last tile touches the edge
    if zs[-1] != Z - pz:
        zs.append(max(0, Z - pz))
    if ys[-1] != Y - py:
        ys.append(max(0, Y - py))
    if xs[-1] != X - px:
        xs.append(max(0, X - px))

    coords = []
    for z0 in zs:
        for y0 in ys:
            for x0 in xs:
                coords.append((z0, z0 + pz, y0, y0 + py, x0, x0 + px))
    return coords


def prep_all(args):
    """Main data preparation pipeline."""
    os.makedirs(os.path.join(args.out_root, "train"), exist_ok=True)
    os.makedirs(os.path.join(args.out_root, "test"), exist_ok=True)

    # Split files by run index
    files = sorted([
        os.path.join(args.input_root, f)
        for f in os.listdir(args.input_root)
        if f.endswith(".h5")
    ])
    runs = sorted([(parse_run_index_from_name(f), f) for f in files], key=lambda x: x[0])

    if len(runs) == 0:
        raise RuntimeError("No .h5 files found in input_root")

    # First 5 indices -> test, rest -> train
    test_runs = set([r for r, _ in runs[:5]])
    train_src = []
    test_src = []
    for r, f in runs:
        if r in test_runs:
            test_src.append(f)
        else:
            train_src.append(f)

    print(f"Found {len(train_src)} training files, {len(test_src)} test files")

    # Pack files
    def pack_many(src_list, split):
        dsts = []
        skipped = []
        for p in tqdm(src_list, desc=f"Packing {split}"):
            base = os.path.basename(p).replace(".h5", ".prepped.h5")
            dst = os.path.join(args.out_root, split, base)
            try:
                pack_one_file(p, dst, dtype=args.dtype, data_format=args.data_format)
                dsts.append(dst)
            except (OSError, Exception) as e:
                print(f"\nSkipping corrupted file {p}: {e}")
                skipped.append(p)
        if skipped:
            print(f"\nWarning: Skipped {len(skipped)} corrupted files in {split} set:")
            for s in skipped:
                print(f"  - {os.path.basename(s)}")
        return dsts

    train_paths = pack_many(train_src, "train")
    test_paths = pack_many(test_src, "test")

    # Compute statistics on training set only
    scalar_map = {name: i for i, name in enumerate(SCALAR_OUTPUT_KEYS)}
    log_transform_set = set([scalar_map[s] for s in args.log_transform])
    stats = compute_stats(train_paths, log_transform_set, data_format=args.data_format)

    stats_path = os.path.join(args.out_root, "stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Saved stats to {stats_path}")

    # Build patch indices
    def write_index(paths, split):
        index_path = os.path.join(args.out_root, f"index_{split}.jsonl")
        with open(index_path, "w") as out:
            for p in tqdm(paths, desc=f"Indexing {split}"):
                with h5py.File(p, "r") as f:
                    T = f["outputs_grid"].shape[0]
                    Z, Y, X = f["outputs_grid"].shape[2:5]

                # Clamp patch size to volume
                pz = min(args.patch_size[0], Z)
                py = min(args.patch_size[1], Y)
                px = min(args.patch_size[2], X)
                sz = max(1, min(args.stride[0], pz))
                sy = max(1, min(args.stride[1], py))
                sx = max(1, min(args.stride[2], px))

                coords = tile_indices((Z, Y, X), (pz, py, px), (sz, sy, sx))

                # Generate (t, t+1) pairs
                for t in range(T - 1):
                    for (z0, z1, y0, y1, x0, x1) in coords:
                        rec = {
                            "sim_path": p,
                            "t": t,
                            "z0": z0, "z1": z1,
                            "y0": y0, "y1": y1,
                            "x0": x0, "x1": x1
                        }
                        out.write(json.dumps(rec) + "\n")
        return index_path

    idx_train = write_index(train_paths, "train")
    idx_test = write_index(test_paths, "test")

    print("\n=== Prep Complete ===")
    print(f"Train files: {len(train_paths)}")
    print(f"Test files: {len(test_paths)}")
    print(f"Stats: {stats_path}")
    print(f"Train index: {idx_train}")
    print(f"Test index: {idx_test}")


def main():
    ap = argparse.ArgumentParser(description="Prepare HDF5 data for training")
    ap.add_argument("--input_root", type=str, required=True, help="Directory with raw v2.4/v2.5 HDF5 files")
    ap.add_argument("--out_root", type=str, required=True, help="Output directory for prepped data")
    ap.add_argument("--data_format", type=str, default="v2.4", choices=["v2.4", "v2.5"],
                    help="Data format version (v2.4 has P,T grid outputs; v2.5 adds WEPT)")
    ap.add_argument("--patch_size", type=int, nargs=3, default=[64, 64, 64], help="Patch size [Z Y X]")
    ap.add_argument("--stride", type=int, nargs=3, default=[64, 64, 64], help="Stride [Z Y X]")
    ap.add_argument("--log_transform", type=str, nargs="*", default=[
        "FieldEnergyInjectionRate",
        "FieldEnergyProductionRate",
        "FieldEnergyProductionTotal",
        "FieldWaterInjectionRate",
        "FieldWaterProductionRate",
    ], help="Scalar fields to apply log1p transform")
    ap.add_argument("--dtype", type=str, default="float32", help="Data type for arrays")
    args = ap.parse_args()
    prep_all(args)


if __name__ == "__main__":
    main()
