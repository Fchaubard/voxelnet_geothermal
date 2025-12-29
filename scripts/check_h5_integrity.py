#!/usr/bin/env python3
"""
Check integrity of raw h5 files and identify corrupted ones.
"""
import glob
import h5py
import os

# Get all raw h5 files
all_files = sorted(glob.glob('/workspace/all_oak_data/h5s_v2.5_data/v2.5_*.h5'))
print(f"Checking {len(all_files)} h5 files for integrity...")

corrupted = []
valid = []

for i, fpath in enumerate(all_files):
    try:
        # Try to open and access data
        with h5py.File(fpath, 'r') as f:
            # Try to access key datasets
            _ = f['Output/Pressure'].shape
            _ = f['Output/Temperature'].shape
            _ = f['Output/WEPT'].shape
        valid.append(fpath)
        if (i + 1) % 50 == 0:
            print(f"  Checked {i+1}/{len(all_files)}: {len(valid)} valid, {len(corrupted)} corrupted")
    except Exception as e:
        corrupted.append((fpath, str(e)))
        print(f"  CORRUPTED: {os.path.basename(fpath)} - {e}")

print(f"\n{'='*80}")
print(f"INTEGRITY CHECK COMPLETE")
print(f"{'='*80}")
print(f"Total files: {len(all_files)}")
print(f"Valid files: {len(valid)}")
print(f"Corrupted files: {len(corrupted)}")

if corrupted:
    print(f"\n{'='*80}")
    print("CORRUPTED FILES:")
    print(f"{'='*80}")
    for fpath, error in corrupted:
        print(f"  {os.path.basename(fpath)}: {error[:100]}")

    # Save list of valid files
    valid_list_path = '/workspace/omv_v2.5/data/valid_raw_h5_files.txt'
    with open(valid_list_path, 'w') as f:
        for fpath in valid:
            f.write(fpath + '\n')
    print(f"\nValid file list saved to: {valid_list_path}")
else:
    print("\nAll files are valid!")
