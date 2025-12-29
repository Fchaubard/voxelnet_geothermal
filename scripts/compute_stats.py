#!/usr/bin/env python3
"""
Compute dual statistics (absolute for inputs, delta for outputs) directly from raw h5 files.
This allows immediate training without waiting for the full prepped data stats recomputation.
"""
import sys
sys.path.insert(0, '/workspace/omv_v2.5')
import glob
import h5py
import numpy as np
import json
import os


class Welford:
    """Online algorithm for computing running mean and variance."""
    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self.M2 = 0.0

    def update(self, new_values):
        """Update with a batch of new values (numpy array)."""
        new_values = np.asarray(new_values).ravel()
        for x in new_values:
            self.count += 1
            delta = x - self.mean
            self.mean += delta / self.count
            delta2 = x - self.mean
            self.M2 += delta * delta2

    def finalize(self):
        """Return (mean, std)."""
        if self.count < 2:
            return self.mean, 0.0
        variance = self.M2 / self.count
        return self.mean, np.sqrt(variance)


def compute_raw_h5_stats(h5_files, num_files=None):
    """
    Compute statistics from raw h5 files.

    Returns stats dict with dual statistics:
    - grid_input_absolute: stats on absolute grid values (for normalizing inputs)
    - grid_output_delta: stats on temporal deltas (for normalizing outputs)
    """
    if num_files:
        h5_files = h5_files[:num_files]

    print(f"Computing stats from {len(h5_files)} raw h5 files...")

    # Initialize Welford accumulators
    # Static channels: FaultId, InjRate, IsActive, IsWell, PermX, PermY, PermZ, Porosity (8 channels - matching RawH5Dataset.STATIC_KEYS)
    static_w = [Welford() for _ in range(8)]

    # Grid channels: Pressure, Temperature, WEPT (3 channels)
    grid_input_w = [Welford() for _ in range(3)]  # For absolute values
    grid_output_w = [Welford() for _ in range(3)]  # For deltas

    # Scalar outputs: 5 channels (we'll compute on absolute/log1p later)
    scalar_w = [Welford() for _ in range(5)]

    for i, fpath in enumerate(h5_files):
        if (i + 1) % 10 == 0:
            print(f"  Processing file {i+1}/{len(h5_files)}...")

        try:
            with h5py.File(fpath, 'r') as f:
                # 1. Static channels (absolute values) - MUST match RawH5Dataset.STATIC_KEYS order
                static_keys = ['FaultId', 'InjRate', 'IsActive', 'IsWell',
                              'PermX', 'PermY', 'PermZ', 'Porosity']
                for ch_idx, key in enumerate(static_keys):
                    if f'Input/{key}' in f:
                        vals = f[f'Input/{key}'][:]
                        valid_mask = (vals != -999)
                        if valid_mask.any():
                            static_w[ch_idx].update(vals[valid_mask])

                # 2. Grid channels - DUAL STATS
                # Get time series data
                pressure = f['Output/Pressure'][:]  # Shape: (T, Z, Y, X)
                temperature = f['Output/Temperature'][:]
                wept = f['Output/WEPT'][:]

                T = pressure.shape[0]
                grids = [pressure, temperature, wept]

                for ch_idx, grid in enumerate(grids):
                    # 2a. Absolute value stats (for normalizing inputs)
                    for t in range(T):
                        vals = grid[t]
                        valid_mask = (vals != -999)
                        if valid_mask.any():
                            grid_input_w[ch_idx].update(vals[valid_mask])

                    # 2b. Delta stats (for normalizing outputs)
                    for t in range(T - 1):
                        curr = grid[t]
                        next_val = grid[t + 1]
                        delta = next_val - curr
                        valid_mask = (curr != -999) & (next_val != -999)
                        if valid_mask.any():
                            grid_output_w[ch_idx].update(delta[valid_mask])

                # 3. Scalar outputs (NO LOG1P - use absolute raw values)
                scalar_keys = [
                    'Metadata/FieldEnergyInjectionRate',
                    'Metadata/FieldEnergyProductionRate',
                    'Metadata/FieldEnergyProductionTotal',
                    'Metadata/FieldWaterInjectionRate',
                    'Metadata/FieldWaterProductionRate'
                ]

                for ch_idx, key in enumerate(scalar_keys):
                    if key in f:
                        vals = f[key][:]
                        # Use absolute raw values, no transforms
                        scalar_w[ch_idx].update(vals)

        except Exception as e:
            print(f"  WARNING: Failed to process {fpath}: {e}")
            continue

    # Finalize statistics
    static_mean, static_std = [], []
    for w in static_w:
        m, s = w.finalize()
        static_mean.append(float(m))
        static_std.append(float(s))

    grid_input_mean, grid_input_std = [], []
    for w in grid_input_w:
        m, s = w.finalize()
        grid_input_mean.append(float(m))
        grid_input_std.append(float(s))

    grid_output_mean, grid_output_std = [], []
    for w in grid_output_w:
        m, s = w.finalize()
        grid_output_mean.append(float(m))
        grid_output_std.append(float(s))

    scalar_mean, scalar_std = [], []
    for w in scalar_w:
        m, s = w.finalize()
        scalar_mean.append(float(m))
        scalar_std.append(float(s))

    # Build stats dictionary
    stats = {
        "mean": {
            "static_input_absolute": static_mean,
            "grid_input_absolute": grid_input_mean,
            "grid_output_delta": grid_output_mean,
            "scalar_output_absolute": scalar_mean,
            # Backward compatibility
            "static": static_mean,
            "grid": grid_output_mean,  # Use delta stats for backward compat
            "scalar": scalar_mean
        },
        "std": {
            "static_input_absolute": static_std,
            "grid_input_absolute": grid_input_std,
            "grid_output_delta": grid_output_std,
            "scalar_output_absolute": scalar_std,
            # Backward compatibility
            "static": static_std,
            "grid": grid_output_std,  # Use delta stats for backward compat
            "scalar": scalar_std
        },
        "static_channels": ["FaultId", "InjRate", "IsActive", "IsWell", "PermX", "PermY", "PermZ", "Porosity"],
        "grid_channels": ["Pressure", "Temperature", "WEPT"],
        "scalar_output_keys": [
            "FieldEnergyInjectionRate",
            "FieldEnergyProductionRate",
            "FieldEnergyProductionTotal",
            "FieldWaterInjectionRate",
            "FieldWaterProductionRate"
        ],
        "log1p_flags": {
            "scalar": [False, False, False, False, False]
        },
        "grid_input_stats_computed_on": "absolute_values",
        "grid_output_stats_computed_on": "temporal_deltas",
        "static_stats_computed_on": "absolute_values",
        "scalar_stats_computed_on": "absolute_values_no_log1p",
        "data_format": "v2.5",
        "num_files_used": len(h5_files)
    }

    return stats


if __name__ == "__main__":
    # Get valid raw h5 files (excluding corrupted ones)
    valid_list_path = '/workspace/omv_v2.5/data/valid_raw_h5_files.txt'

    if os.path.exists(valid_list_path):
        with open(valid_list_path, 'r') as f:
            all_files = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(all_files)} valid raw h5 files from {valid_list_path}")
    else:
        all_files = sorted(glob.glob('/workspace/all_oak_data/h5s_v2.5_data/v2.5_*.h5'))
        print(f"Found {len(all_files)} total raw h5 files (no valid list found)")

    # Use first 80% for training stats (matching RawH5Dataset split)
    n_train = int(len(all_files) * 0.8)
    train_files = all_files[:n_train]
    print(f"Using {len(train_files)} training files for stats")

    # Compute stats on subset for speed (use 100 files for good coverage)
    num_files = len(train_files)
    stats = compute_raw_h5_stats(train_files, num_files=num_files)

    # Save stats
    stats_path = '/workspace/omv_v2.5/data/raw_h5_stats_full.json'
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    # Print verification
    print(f'\n{"="*80}')
    print('STATISTICS SUMMARY')
    print(f'{"="*80}')

    print('\n=== Grid INPUT Statistics (ABSOLUTE VALUES) ===')
    for ch_name, mean, std in zip(stats['grid_channels'],
                                    stats['mean']['grid_input_absolute'],
                                    stats['std']['grid_input_absolute']):
        print(f'{ch_name:12s}: mean={mean:15.2f}, std={std:15.2f}')

    print('\n=== Grid OUTPUT Statistics (TEMPORAL DELTAS) ===')
    for ch_name, mean, std in zip(stats['grid_channels'],
                                    stats['mean']['grid_output_delta'],
                                    stats['std']['grid_output_delta']):
        print(f'{ch_name:12s}: mean={mean:15.6f}, std={std:15.6f}')

    print('\n=== Static Channel Statistics (ABSOLUTE VALUES) ===')
    for ch_name, mean, std in zip(stats['static_channels'],
                                    stats['mean']['static'],
                                    stats['std']['static']):
        print(f'{ch_name:12s}: mean={mean:15.2f}, std={std:15.2f}')

    print(f'\n{"="*80}')
    print(f'Stats saved to: {stats_path}')
    print(f'Computed from {stats["num_files_used"]} training files')
    print(f'Grid input stats: {stats["grid_input_stats_computed_on"]}')
    print(f'Grid output stats: {stats["grid_output_stats_computed_on"]}')
    print(f'{"="*80}')

    # Verify no zero or near-zero std values
    print('\n=== VALIDATION ===')
    all_stds = (stats['std']['grid_input_absolute'] +
                stats['std']['grid_output_delta'] +
                stats['std']['static'] +
                stats['std']['scalar'])

    min_std = min(all_stds)
    if min_std < 1e-6:
        print(f'WARNING: Found near-zero std: {min_std:.6e}')
    else:
        print(f'✓ All std values are reasonable (min={min_std:.6e})')
