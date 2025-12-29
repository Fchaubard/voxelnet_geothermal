"""PyTorch dataset for voxel grid patches with augmentations."""
import datetime
import json
import os
import h5py
import torch
import numpy as np
import threading
import queue
import time
from typing import Any, Dict, Optional
from torch.utils.data import Dataset


def pad_collate_fn(batch, fixed_size=None):
    """
    Custom collate function that pads tensors to handle variable shapes from rotation augmentations.

    When 90° or 270° rotations are applied to non-square grids, Y and X dimensions swap.
    This function pads all tensors to the max dimensions in the batch.

    Args:
        batch: List of data samples
        fixed_size: Optional (Z, Y, X) tuple to pad to fixed size (prevents fragmentation)
    """
    if fixed_size is not None:
        max_z, max_y, max_x = fixed_size
    else:
        # Find max dimensions across batch (original behavior)
        max_z = max(item["x_grid"].shape[1] for item in batch)
        max_y = max(item["x_grid"].shape[2] for item in batch)
        max_x = max(item["x_grid"].shape[3] for item in batch)

    # Pad each item
    x_grids = []
    y_grids = []
    params = []
    y_scalars = []
    valid_masks = []

    for item in batch:
        x = item["x_grid"]
        y = item["y_grid"]
        mask = item["valid_mask"]

        # Compute padding needed (pad only on right/bottom)
        pad_z = max_z - x.shape[1]
        pad_y = max_y - x.shape[2]
        pad_x = max_x - x.shape[3]

        # Pad format for 4D tensors: (left, right, top, bottom, front, back)
        padding_4d = (0, pad_x, 0, pad_y, 0, pad_z)
        # Pad format for 3D tensors: (left, right, top, bottom, front, back)
        padding_3d = (0, pad_x, 0, pad_y, 0, pad_z)

        x_padded = torch.nn.functional.pad(x, padding_4d, mode='constant', value=0)
        y_padded = torch.nn.functional.pad(y, padding_4d, mode='constant', value=0)
        mask_padded = torch.nn.functional.pad(mask, padding_3d, mode='constant', value=0)  # Pad with 0 (invalid)

        x_grids.append(x_padded)
        y_grids.append(y_padded)
        valid_masks.append(mask_padded)
        params.append(item["params"])
        y_scalars.append(item["y_scalar"])

    return {
        "x_grid": torch.stack(x_grids),
        "params": torch.stack(params),
        "y_grid": torch.stack(y_grids),
        "y_scalar": torch.stack(y_scalars),
        "valid_mask": torch.stack(valid_masks),
    }


class InMemoryH5File:
    """Lightweight wrapper around a fully materialized H5 tree."""

    def __init__(self, tree: Dict[str, Any], path: str):
        self.tree = tree
        self.path = path

    def __getitem__(self, key: str):
        if not isinstance(key, str):
            raise KeyError(f"H5Cache: unsupported key type {type(key)}")
        node = self.tree
        for part in key.split('/'):
            if not part:
                continue
            node = node[part]
        return node

    def close(self):
        self.tree = {}


class H5Cache:
    """Progressively loads raw H5 files into RAM with background prefetching."""

    def __init__(
        self,
        max_files: int = 150,
        file_list=None,
        load_into_memory: bool = False,
        async_prefetch: bool = False,
        load_workers: int = 1,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.handles: Dict[str, Any] = {}
        self.load_order = []
        self.max_files = max_files
        self.file_list = file_list or []
        self.rank = rank
        self.world_size = world_size
        # Start at rank offset so each rank loads different files
        self.next_file_idx = rank
        self.load_into_memory = load_into_memory
        self.async_prefetch = async_prefetch and load_into_memory
        self.load_workers = max(1, load_workers) if self.async_prefetch else 0

        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
        self.pending = set()
        self.failed_files = set()  # Track corrupted/truncated files to skip
        self.stop_event = threading.Event()
        self.queue = queue.Queue()
        self.loader_threads = []
        if self.async_prefetch:
            for _ in range(self.load_workers):
                t = threading.Thread(target=self._loader_loop, daemon=True)
                t.start()
                self.loader_threads.append(t)

    def _copy_dataset(self, dataset):
        arr = dataset[...]
        if np.issubdtype(arr.dtype, np.floating):
            arr = arr.astype(np.float32)
        elif np.issubdtype(arr.dtype, np.integer):
            arr = arr.astype(np.int32)
        else:
            arr = np.array(arr)
        return arr

    def _copy_group(self, group):
        tree = {}
        for name, item in group.items():
            if isinstance(item, h5py.Dataset):
                tree[name] = self._copy_dataset(item)
            elif isinstance(item, h5py.Group):
                tree[name] = self._copy_group(item)
        return tree

    def _materialize_file(self, path: str):
        with h5py.File(path, "r") as h5:
            tree = {}
            for name, item in h5.items():
                if isinstance(item, h5py.Dataset):
                    tree[name] = self._copy_dataset(item)
                elif isinstance(item, h5py.Group):
                    tree[name] = self._copy_group(item)
            return InMemoryH5File(tree, path)

    def _evict_oldest_locked(self):
        if not self.load_order:
            return
        oldest_path = self.load_order.pop(0)
        handle = self.handles.pop(oldest_path, None)
        if handle is not None:
            try:
                handle.close()
            except Exception:
                pass

    def _loader_loop(self):
        while not self.stop_event.is_set():
            try:
                path = self.queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                self._load_file(path)
            finally:
                self.queue.task_done()

    def _load_file(self, path: str):
        start = time.time()
        try:
            handle = self._materialize_file(path) if self.load_into_memory else h5py.File(path, "r")
        except Exception as e:
            print(f"H5Cache: Error loading {path}: {e} - marking as failed, will skip in future")
            with self.condition:
                self.pending.discard(path)
                self.failed_files.add(path)  # Never retry this corrupted file
            return False

        with self.condition:
            if path in self.handles:
                # Already loaded by another worker
                try:
                    handle.close()
                except Exception:
                    pass
                self.pending.discard(path)
                return False

            if len(self.handles) >= self.max_files:
                self._evict_oldest_locked()

            self.handles[path] = handle
            self.load_order.append(path)
            self.pending.discard(path)
            self.condition.notify_all()

        elapsed = time.time() - start
        basename = os.path.basename(path)
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] H5Cache (rank{self.rank}): Loaded {basename} into RAM in {elapsed:.1f}s ({len(self.handles)}/{self.max_files})")
        return True

    def preload_initial_files(self, num_files: int = 1):
        loaded = 0
        target = min(num_files, len(self.file_list))
        while loaded < target and self.next_file_idx < len(self.file_list):
            path = self.file_list[self.next_file_idx]
            # Stride by world_size so each rank loads different files
            self.next_file_idx += self.world_size
            if path in self.handles or path in self.failed_files:
                continue
            if self._load_file(path):
                loaded += 1
        return loaded

    def request_background_load(self, num_files: int = 1):
        if not self.file_list:
            return 0
        if not self.async_prefetch:
            return self.preload_initial_files(num_files)

        requested = 0
        attempts = 0
        max_attempts = len(self.file_list)  # Prevent infinite loop

        with self.condition:
            while requested < num_files and attempts < max_attempts:
                # Wrap around when we reach the end to cycle through all files
                if self.next_file_idx >= len(self.file_list):
                    self.next_file_idx = self.rank  # Reset to starting offset

                path = self.file_list[self.next_file_idx]
                # Stride by world_size so each rank loads different files
                self.next_file_idx += self.world_size
                attempts += 1

                # Skip if already in cache, pending, or failed
                if path in self.handles or path in self.pending or path in self.failed_files:
                    continue

                self.pending.add(path)
                self.queue.put(path)
                requested += 1
        return requested

    def load_next_file(self):
        return self.request_background_load(1) > 0

    def get(self, path):
        with self.condition:
            handle = self.handles.get(path)
        if handle is not None:
            return handle

        # Lazy-load if not part of progressive list or not yet queued
        if not self.load_into_memory:
            if self._load_file(path):
                with self.condition:
                    return self.handles.get(path)
        return None

    def get_loaded_files(self):
        with self.condition:
            return list(self.load_order)

    def wait_for_min_files(self, count: int, timeout: Optional[float] = None):
        deadline = None if timeout is None else time.time() + timeout
        with self.condition:
            while len(self.load_order) < count:
                remaining = None if deadline is None else deadline - time.time()
                if remaining is not None and remaining <= 0:
                    return False
                self.condition.wait(timeout=remaining)
            return True

    def close(self):
        self.stop_event.set()
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break
        for t in self.loader_threads:
            t.join(timeout=0.1)
        with self.condition:
            for handle in self.handles.values():
                try:
                    handle.close()
                except Exception:
                    pass
            self.handles.clear()
            self.load_order.clear()

    def __del__(self):
        self.close()


class VoxelARIndexDataset(Dataset):
    """
    Dataset for autoregressive voxel prediction from patch indices.

    Loads:
        - Static features (geology, wells, etc.)
        - Grid state at time t-1 (Pressure, Temperature, WEPT) - or zeros if t < 2
        - Grid state at time t (Pressure, Temperature, WEPT)
        - Targets at time t+1 as RESIDUALS (delta from t to t+1)

    Applies:
        - Per-channel standardization
        - Optional log1p for scalars
        - XY augmentations (rotation, flip) - never Z axis
        - Gaussian noise on inputs
    """

    def __init__(
        self,
        index_path: str,
        stats_path: str,
        augment: bool = True,
        aug_xy_rot: bool = True,
        aug_flip: bool = True,
        noise_std: float = 0.0,
        noise_exclude_static_idx=(0, 2, 3),  # FaultId, IsActive, IsWell
        use_param_broadcast: bool = False,
        use_params_as_condition: bool = True,
        predict_grid_only: bool = False,
        predict_residuals: bool = True,  # Predict deltas instead of absolute values
    ):
        super().__init__()

        # Load index
        self.index = []
        with open(index_path, "r") as f:
            for line in f:
                self.index.append(json.loads(line))

        # Load statistics
        with open(stats_path, "r") as f:
            self.stats = json.load(f)

        self.augment = augment
        self.aug_xy_rot = aug_xy_rot
        self.aug_flip = aug_flip
        self.noise_std = noise_std
        self.noise_exclude_static_idx = set(noise_exclude_static_idx)
        self.use_param_broadcast = use_param_broadcast
        self.use_params_as_condition = use_params_as_condition
        self.predict_grid_only = predict_grid_only
        self.predict_residuals = predict_residuals
        self.h5cache = H5Cache()

        # Determine data format
        self.data_format = self.stats.get("data_format", "v2.4")
        self.num_grid_channels = len(self.stats["grid_channels"])

        # Convert stats to tensors
        self.static_mean = torch.tensor(self.stats["mean"]["static"], dtype=torch.float32)
        self.static_std = torch.tensor(self.stats["std"]["static"], dtype=torch.float32)
        self.grid_mean = torch.tensor(self.stats["mean"]["grid"], dtype=torch.float32)
        self.grid_std = torch.tensor(self.stats["std"]["grid"], dtype=torch.float32)
        self.scalar_mean = torch.tensor(self.stats["mean"]["scalar"], dtype=torch.float32)
        self.scalar_std = torch.tensor(self.stats["std"]["scalar"], dtype=torch.float32)
        self.scalar_log1p_flags = self.stats["log1p_flags"]["scalar"]

    def __len__(self):
        return len(self.index)

    def _standardize_static(self, static):
        """Standardize static channels: [C_static, Z, Y, X]

        Replaces -999 sentinels with 0 after normalization to avoid large negative values.
        """
        x = torch.from_numpy(static).float()
        m = self.static_mean.view(-1, 1, 1, 1)
        s = self.static_std.view(-1, 1, 1, 1).clamp(min=1e-6)
        normalized = (x - m) / s
        # Replace sentinels with 0 (neutral value, won't bias model)
        normalized[x == -999] = 0
        return normalized

    def _standardize_grid(self, grid, ch, subtract_mean=True):
        """Standardize grid channel (0=Pressure, 1=Temperature, 2=WEPT): [Z, Y, X]

        Replaces -999 sentinels with 0 after normalization to avoid large negative values.

        Args:
            grid: Input grid array
            ch: Channel index
            subtract_mean: If False, only divide by std (for normalizing deltas)
        """
        x = torch.from_numpy(grid).float()
        s = self.grid_std[ch].clamp(min=1e-6)
        if subtract_mean:
            m = self.grid_mean[ch]
            normalized = (x - m) / s
        else:
            # For deltas: only scale by std, don't subtract mean
            normalized = x / s

        # Replace sentinels with 0 (neutral value, won't bias model)
        normalized[x == -999] = 0
        return normalized

    def _standardize_scalar(self, val, ch):
        """Standardize scalar (with optional log1p): scalar value"""
        x = torch.tensor(val, dtype=torch.float32)
        if self.scalar_log1p_flags[ch]:
            x = torch.log1p(torch.clamp(x, min=0))
        m = self.scalar_mean[ch]
        s = self.scalar_std[ch].clamp(min=1e-6)
        return (x - m) / s

    def __getitem__(self, idx):
        rec = self.index[idx]
        f = self.h5cache.get(rec["sim_path"])
        t = rec["t"]
        z0, z1, y0, y1, x0, x1 = rec["z0"], rec["z1"], rec["y0"], rec["y1"], rec["x0"], rec["x1"]

        # Get spatial dimensions
        pz, py, px = z1 - z0, y1 - y0, x1 - x0

        # Load data
        static = f["static"][:, z0:z1, y0:y1, x0:x1]  # [C_static, pz, py, px]
        params = f["params_scalar"][...]  # [26]

        # Load t-1 data (or zeros if t < 2)
        if t >= 2:
            p_tm1 = f["outputs_grid"][t - 1, 0, z0:z1, y0:y1, x0:x1]
            T_tm1 = f["outputs_grid"][t - 1, 1, z0:z1, y0:y1, x0:x1]
            if self.data_format == "v2.5" and not self.predict_grid_only and self.num_grid_channels >= 3:
                wept_tm1 = f["outputs_grid"][t - 1, 2, z0:z1, y0:y1, x0:x1]
        else:
            # t < 2: use zeros for t-1 (will be zeros after normalization)
            p_tm1 = np.zeros((pz, py, px), dtype=np.float32)
            T_tm1 = np.zeros((pz, py, px), dtype=np.float32)
            if self.data_format == "v2.5" and not self.predict_grid_only and self.num_grid_channels >= 3:
                wept_tm1 = np.zeros((pz, py, px), dtype=np.float32)

        # Load t data
        p_t = f["outputs_grid"][t, 0, z0:z1, y0:y1, x0:x1]
        T_t = f["outputs_grid"][t, 1, z0:z1, y0:y1, x0:x1]
        if self.data_format == "v2.5" and not self.predict_grid_only and self.num_grid_channels >= 3:
            if t >= 1:
                wept_t = f["outputs_grid"][t, 2, z0:z1, y0:y1, x0:x1]
            else:
                # t == 0: WEPT is 0 at t=0
                wept_t = np.zeros((pz, py, px), dtype=np.float32)

        # Standardize inputs
        static = self._standardize_static(static)  # [C_static, ...]

        # Standardize t-1
        p_tm1 = self._standardize_grid(p_tm1, 0).unsqueeze(0)
        T_tm1 = self._standardize_grid(T_tm1, 1).unsqueeze(0)

        # Standardize t
        p_t = self._standardize_grid(p_t, 0).unsqueeze(0)
        T_t = self._standardize_grid(T_t, 1).unsqueeze(0)

        # Build input grid: [static, P_{t-1}, T_{t-1}, WEPT_{t-1}, P_t, T_t, WEPT_t]
        input_grids = [static, p_tm1, T_tm1]

        # Add WEPT_{t-1} for v2.5
        if self.data_format == "v2.5" and not self.predict_grid_only and self.num_grid_channels >= 3:
            wept_tm1 = self._standardize_grid(wept_tm1, 2).unsqueeze(0)
            input_grids.append(wept_tm1)

        # Add t data
        input_grids.extend([p_t, T_t])

        # Add WEPT_t for v2.5
        if self.data_format == "v2.5" and not self.predict_grid_only and self.num_grid_channels >= 3:
            wept_t = self._standardize_grid(wept_t, 2).unsqueeze(0)
            input_grids.append(wept_t)

        x_grid = torch.cat(input_grids, dim=0).contiguous()

        # Scalar parameters
        params_t = torch.from_numpy(params).float()
        params_condition = params_t.clone()

        if self.use_param_broadcast:
            # Broadcast params as additional channels
            Bc = params_t.view(-1, 1, 1, 1).expand(-1, x_grid.shape[1], x_grid.shape[2], x_grid.shape[3])
            x_grid = torch.cat([x_grid, Bc], dim=0)

        # Targets at t+1
        p_tp1_raw = f["outputs_grid"][t + 1, 0, z0:z1, y0:y1, x0:x1]
        T_tp1_raw = f["outputs_grid"][t + 1, 1, z0:z1, y0:y1, x0:x1]
        scalars_tp1 = f["outputs_scalar"][t + 1, :]  # [5]

        if self.predict_residuals:
            # Predict residuals: delta = (t+1) - t
            # Load t data in raw form for residual computation
            p_t_raw = f["outputs_grid"][t, 0, z0:z1, y0:y1, x0:x1]
            T_t_raw = f["outputs_grid"][t, 1, z0:z1, y0:y1, x0:x1]

            # Compute deltas
            delta_p = p_tp1_raw - p_t_raw
            delta_T = T_tp1_raw - T_t_raw

            # Standardize deltas (don't subtract mean for deltas, only scale by std)
            target_grids = [
                self._standardize_grid(delta_p, 0, subtract_mean=False),
                self._standardize_grid(delta_T, 1, subtract_mean=False),
            ]

            # In v2.5, also predict WEPT delta
            if self.data_format == "v2.5" and not self.predict_grid_only and self.num_grid_channels >= 3:
                wept_tp1_raw = f["outputs_grid"][t + 1, 2, z0:z1, y0:y1, x0:x1]
                if t >= 1:
                    wept_t_raw = f["outputs_grid"][t, 2, z0:z1, y0:y1, x0:x1]
                else:
                    # t == 0: WEPT_0 is 0
                    wept_t_raw = np.zeros((pz, py, px), dtype=np.float32)
                delta_wept = wept_tp1_raw - wept_t_raw
                target_grids.append(self._standardize_grid(delta_wept, 2, subtract_mean=False))
        else:
            # Predict absolute values (legacy behavior)
            target_grids = [
                self._standardize_grid(p_tp1_raw, 0),
                self._standardize_grid(T_tp1_raw, 1),
            ]

            # In v2.5, when not predict_grid_only, also predict WEPT
            if self.data_format == "v2.5" and not self.predict_grid_only and self.num_grid_channels >= 3:
                wept_tp1_raw = f["outputs_grid"][t + 1, 2, z0:z1, y0:y1, x0:x1]
                target_grids.append(self._standardize_grid(wept_tp1_raw, 2))

        y_grid = torch.stack(target_grids, dim=0)

        y_scalar = torch.stack([
            self._standardize_scalar(scalars_tp1[i], i) for i in range(5)
        ], dim=0)

        # Create validity mask for targets (mask -999 sentinel values)
        # Mask is True for valid cells, False for -999 cells
        valid_mask = torch.from_numpy((p_tp1_raw != -999) & (T_tp1_raw != -999)).float()

        # Augmentations (XY only, never Z)
        if self.augment:
            # Add batch dim for rotation
            x_aug = x_grid.unsqueeze(0)
            y_aug = y_grid.unsqueeze(0)
            mask_aug = valid_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, Z, Y, X]

            # Random 90° rotation in XY plane (dims -2, -1 are Y, X)
            k = np.random.randint(0, 4) if self.aug_xy_rot else 0
            if k > 0:
                x_aug = torch.rot90(x_aug, k, dims=(-2, -1))
                y_aug = torch.rot90(y_aug, k, dims=(-2, -1))
                mask_aug = torch.rot90(mask_aug, k, dims=(-2, -1))

            # Random flips in X and Y (never Z)
            if self.aug_flip and (np.random.rand() < 0.5):
                x_aug = torch.flip(x_aug, dims=[-1])  # flip X
                y_aug = torch.flip(y_aug, dims=[-1])
                mask_aug = torch.flip(mask_aug, dims=[-1])
            if self.aug_flip and (np.random.rand() < 0.5):
                x_aug = torch.flip(x_aug, dims=[-2])  # flip Y
                y_aug = torch.flip(y_aug, dims=[-2])
                mask_aug = torch.flip(mask_aug, dims=[-2])

            x_grid = x_aug.squeeze(0)
            y_grid = y_aug.squeeze(0)
            valid_mask = mask_aug.squeeze(0).squeeze(0)  # [Z, Y, X]

            # Gaussian noise on inputs (exclude categorical channels)
            if self.noise_std > 0:
                noisy = x_grid.clone()
                for i in range(x_grid.shape[0]):
                    if i < static.shape[0] and i in self.noise_exclude_static_idx:
                        continue
                    noisy[i] = noisy[i] + torch.randn_like(noisy[i]) * self.noise_std
                x_grid = noisy

        return {
            "x_grid": x_grid.float(),  # [C_in, Z, Y, X]
            "params": params_condition.float(),  # [26]
            "y_grid": y_grid.float(),  # [2 or 3, Z, Y, X]
            "y_scalar": y_scalar.float(),  # [5]
            "valid_mask": valid_mask.float(),  # [Z, Y, X] - 1.0 for valid, 0.0 for -999
        }


class RawH5Dataset(Dataset):
    """
    Dataset that reads directly from raw h5 files (v2.5 format only).

    Avoids the preprocessing step - loads data on-the-fly from:
        /workspace/all_oak_data/h5s_v2.5_data/v2.5_XXXX.h5

    Generates training samples from full grids (no patching) for each timestep.
    Uses gradient accumulation across multiple GPUs to achieve macrobatch diversity.

    Args:
        raw_h5_dir: Directory containing raw v2.5_*.h5 files
        stats_path: Path to stats.json for normalization
        split: "train" or "test" - uses 80/20 split based on file index
        augment: Enable XY augmentations (rotation, flip)
        noise_std: Gaussian noise std for inputs
        predict_residuals: Predict deltas instead of absolute values
    """

    def __init__(
        self,
        raw_h5_dir: str,
        stats_path: str,
        split: str = "train",  # "train" or "test"
        augment: bool = True,
        aug_xy_rot: bool = True,
        aug_flip: bool = True,
        noise_std: float = 0.0,
        noise_exclude_static_idx=(0, 2, 3),  # FaultId, IsActive, IsWell
        predict_residuals: bool = True,
        files_in_memory: int = 150,  # Number of H5 files to keep cached (~195GB RAM, 34% of dataset)
        virtual_dataset_size: int = 100000,  # Virtual dataset size for sampling
        initial_preload: int = 1,
        cache_prefetch_steps: Optional[int] = 5,
        async_prefetch: bool = True,
        cache_load_workers: int = 1,
    ):
        super().__init__()

        import glob
        import os
        import re

        # Load valid files list (excludes corrupted files)
        valid_files_path = "/workspace/omv_v2.5/data/valid_raw_h5_files.txt"
        if os.path.exists(valid_files_path):
            with open(valid_files_path, 'r') as f:
                all_files = sorted([line.strip() for line in f if line.strip()])
            print(f"RawH5Dataset: Loaded {len(all_files)} valid files from {valid_files_path}")
        else:
            # Fallback to globbing all files
            all_files = sorted(glob.glob(os.path.join(raw_h5_dir, "v2.5_*.h5")))
            print(f"RawH5Dataset: Warning - valid_raw_h5_files.txt not found, using all files")

        if len(all_files) == 0:
            raise ValueError(f"No v2.5_*.h5 files found in {raw_h5_dir}")

        # Split by file number: v2.5_0001 to v2.5_0005 = test, v2.5_0006+ = train
        test_files = []
        train_files = []
        for fpath in all_files:
            fname = os.path.basename(fpath)
            match = re.search(r'v2\.5_(\d{4})\.h5', fname)
            if match:
                file_num = int(match.group(1))
                if file_num <= 5:
                    test_files.append(fpath)
                else:
                    train_files.append(fpath)
            else:
                # If no match, assume train (backward compatibility)
                train_files.append(fpath)

        if split == "train":
            self.file_paths = train_files
        else:
            self.file_paths = test_files

        print(f"RawH5Dataset ({split}): Found {len(self.file_paths)} files (test=001-005, train=006+)")

        # Memory-efficient: don't build full sample index upfront
        # Instead, we'll sample randomly in __getitem__
        self.files_in_memory = files_in_memory
        self.virtual_dataset_size = virtual_dataset_size

        # Estimate average timesteps per file for sampling (assume ~28 timesteps on average)
        # Valid t range is [1, T-2] so approximately T-2 valid timesteps per file
        self.avg_timesteps_per_file = 26  # Conservative estimate

        # Load statistics
        with open(stats_path, "r") as f:
            self.stats = json.load(f)

        self.augment = augment
        self.aug_xy_rot = aug_xy_rot
        self.aug_flip = aug_flip
        self.noise_std = noise_std
        self.noise_exclude_static_idx = set(noise_exclude_static_idx)
        self.predict_residuals = predict_residuals
        self.split = split
        self.cache_prefetch_steps = cache_prefetch_steps if cache_prefetch_steps and cache_prefetch_steps > 0 else None
        self._last_cache_prefetch_step = -self.cache_prefetch_steps if self.cache_prefetch_steps else 0

        # Get DDP rank and world_size to distribute file loading across ranks
        # Only use rank-aware loading for train split (test has only 5 files, not enough for all ranks)
        rank = 0
        world_size = 1
        if split == "train" and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()

        # Progressive cache loading: fully materialize files into RAM
        # Each rank loads different files (stride by world_size) - only for train split
        self.h5cache = H5Cache(
            max_files=files_in_memory,
            file_list=self.file_paths,
            load_into_memory=True,
            async_prefetch=async_prefetch,
            load_workers=cache_load_workers,
            rank=rank,
            world_size=world_size,
        )

        preload_target = max(1, min(initial_preload, len(self.file_paths)))
        num_preloaded = self.h5cache.preload_initial_files(num_files=preload_target)
        print(f"RawH5Dataset ({split}): Preloaded {num_preloaded} files into cache (target={preload_target})")
        print(f"RawH5Dataset ({split}): Progressive cache capacity = {files_in_memory} files")

        if async_prefetch and len(self.file_paths) > num_preloaded:
            self.h5cache.request_background_load(1)
            print(
                f"RawH5Dataset ({split}): Async prefetch enabled (load every {self.cache_prefetch_steps or 'manual'} steps)"
            )

        # Convert stats to tensors (use grid_input_absolute for inputs if available)
        self.static_mean = torch.tensor(self.stats["mean"]["static"], dtype=torch.float32)
        self.static_std = torch.tensor(self.stats["std"]["static"], dtype=torch.float32)

        # Use grid_input_absolute for normalizing inputs if available (new dual stats)
        if "grid_input_absolute" in self.stats["mean"]:
            self.grid_input_mean = torch.tensor(self.stats["mean"]["grid_input_absolute"], dtype=torch.float32)
            self.grid_input_std = torch.tensor(self.stats["std"]["grid_input_absolute"], dtype=torch.float32)
        else:
            # Fallback to old "grid" key
            self.grid_input_mean = torch.tensor(self.stats["mean"]["grid"], dtype=torch.float32)
            self.grid_input_std = torch.tensor(self.stats["std"]["grid"], dtype=torch.float32)

        # Use grid_output_delta for normalizing outputs if available (new dual stats)
        if "grid_output_delta" in self.stats["mean"]:
            self.grid_output_mean = torch.tensor(self.stats["mean"]["grid_output_delta"], dtype=torch.float32)
            self.grid_output_std = torch.tensor(self.stats["std"]["grid_output_delta"], dtype=torch.float32)
        else:
            # Fallback to old "grid" key
            self.grid_output_mean = torch.tensor(self.stats["mean"]["grid"], dtype=torch.float32)
            self.grid_output_std = torch.tensor(self.stats["std"]["grid"], dtype=torch.float32)

        self.scalar_mean = torch.tensor(self.stats["mean"]["scalar"], dtype=torch.float32)
        self.scalar_std = torch.tensor(self.stats["std"]["scalar"], dtype=torch.float32)
        self.scalar_log1p_flags = self.stats["log1p_flags"]["scalar"]

        # Static keys ordering from data_prep.py
        self.STATIC_KEYS = [
            "FaultId", "InjRate", "IsActive", "IsWell",
            "PermX", "PermY", "PermZ", "Porosity",
        ]

    def maybe_trigger_cache_load(self, global_step: int):
        if self.cache_prefetch_steps is None:
            return
        if (global_step - self._last_cache_prefetch_step) < self.cache_prefetch_steps:
            return
        loaded = self.h5cache.request_background_load(1)
        if loaded:
            self._last_cache_prefetch_step = global_step

    def __len__(self):
        return self.virtual_dataset_size

    def __del__(self):
        try:
            self.h5cache.close()
        except Exception:
            pass

    def _standardize_static(self, static):
        """Standardize static channels: [C_static, Z, Y, X]

        Replaces -999 sentinels with 0 after normalization to avoid large negative values.
        """
        x = torch.from_numpy(static).float()
        m = self.static_mean.view(-1, 1, 1, 1)
        s = self.static_std.view(-1, 1, 1, 1).clamp(min=1e-6)
        normalized = (x - m) / s
        # Replace sentinels with 0 (neutral value, won't bias model)
        normalized[x == -999] = 0
        return normalized

    def _standardize_grid_input(self, grid, ch):
        """Standardize grid channel for INPUT: [Z, Y, X]

        Uses grid_input_absolute stats (large absolute values).
        Replaces -999 sentinels with 0 after normalization.
        """
        x = torch.from_numpy(grid).float()
        m = self.grid_input_mean[ch]
        s = self.grid_input_std[ch].clamp(min=1e-6)
        normalized = (x - m) / s
        # Replace sentinels with 0
        normalized[x == -999] = 0
        return normalized

    def _standardize_grid_output(self, grid, ch):
        """Standardize grid channel for OUTPUT (delta): [Z, Y, X]

        Uses grid_output_delta stats (small delta values).
        Only scales by std, does not subtract mean (deltas are centered at 0).
        Replaces -999 sentinels with 0 after normalization.
        """
        x = torch.from_numpy(grid).float()
        s = self.grid_output_std[ch].clamp(min=1e-6)
        # For deltas: only scale by std, don't subtract mean
        normalized = x / s
        # Replace sentinels with 0
        normalized[x == -999] = 0
        return normalized

    def _standardize_scalar(self, val, ch):
        """Standardize scalar (with optional log1p): scalar value"""
        x = torch.tensor(val, dtype=torch.float32)
        if self.scalar_log1p_flags[ch]:
            x = torch.log1p(torch.clamp(x, min=0))
        m = self.scalar_mean[ch]
        s = self.scalar_std[ch].clamp(min=1e-6)
        return (x - m) / s

    def __getitem__(self, idx):
        # Progressive cache loading: sample only from currently loaded files
        import random
        rng = random.Random(idx)

        # Get currently loaded files (no waiting - proceed once files are available)
        loaded_files = self.h5cache.get_loaded_files()
        if not loaded_files:
            # No files loaded yet - sleep briefly and let dataloader retry
            import time
            time.sleep(0.01)
            loaded_files = self.h5cache.get_loaded_files()
            if not loaded_files:
                raise RuntimeError("H5Cache: No files loaded yet, dataloader will retry")

        # Randomly select from loaded files only
        file_path = rng.choice(loaded_files)
        f = self.h5cache.get(file_path)
        if f is None:
            # Race condition: file was in list but not available - retry with different file
            loaded_files = self.h5cache.get_loaded_files()
            if not loaded_files:
                raise RuntimeError("H5Cache: No files available")
            file_path = rng.choice(loaded_files)
            f = self.h5cache.get(file_path)
            if f is None:
                raise RuntimeError(f"H5Cache: File {file_path} missing from cache")

        # Get actual number of timesteps in this file
        T = f["Output/Pressure"].shape[0]

        # Randomly select a valid timestep t in range [0, T-2]
        # (need t, t+1 so t must be in [0, T-2]; t-1 uses zeros when t=0)
        if T < 2:
            # Edge case: file has < 2 timesteps, skip to next
            raise RuntimeError(f"File {file_path} has < 2 timesteps")

        t = rng.randint(0, T - 2)  # Include t=0 for first-step prediction training

        # Load static fields from Input/ group
        static_list = []
        for key in self.STATIC_KEYS:
            arr = f[f"Input/{key}"][...]  # [Z, Y, X]
            static_list.append(arr)
        static = np.stack(static_list, axis=0)  # [C_static, Z, Y, X]

        # Load scalar parameters
        params = f["Input/ParamsScalar"][...]  # [26]

        # Load grid data at t-1, t, t+1 from Output/ group
        # Output structure: Pressure, Temperature, WEPT all have shape (T, Z, Y, X)
        # When t=0, use zeros for t-1 (matches inference rollout initialization)
        if t >= 1:
            p_tm1 = f["Output/Pressure"][t - 1, ...]  # [Z, Y, X]
            T_tm1 = f["Output/Temperature"][t - 1, ...]
            wept_tm1 = f["Output/WEPT"][t - 1, ...]
        else:
            # t=0: use zeros for t-1 (no t=-1 exists, matches inference start)
            grid_shape = f["Output/Pressure"].shape[1:]  # (Z, Y, X)
            p_tm1 = np.zeros(grid_shape, dtype=np.float32)
            T_tm1 = np.zeros(grid_shape, dtype=np.float32)
            wept_tm1 = np.zeros(grid_shape, dtype=np.float32)

        p_t = f["Output/Pressure"][t, ...]
        T_t = f["Output/Temperature"][t, ...]
        wept_t = f["Output/WEPT"][t, ...]

        p_tp1 = f["Output/Pressure"][t + 1, ...]
        T_tp1 = f["Output/Temperature"][t + 1, ...]
        wept_tp1 = f["Output/WEPT"][t + 1, ...]

        # Load scalar outputs at t+1
        scalars_tp1 = np.array([
            f["Output/FieldEnergyInjectionRate"][t + 1],
            f["Output/FieldEnergyProductionRate"][t + 1],
            f["Output/FieldEnergyProductionTotal"][t + 1],
            f["Output/FieldWaterInjectionRate"][t + 1],
            f["Output/FieldWaterProductionRate"][t + 1],
        ], dtype=np.float32)

        # Standardize static inputs
        static = self._standardize_static(static)  # [C_static, Z, Y, X]

        # Standardize grid inputs using grid_input_absolute stats
        p_tm1 = self._standardize_grid_input(p_tm1, 0).unsqueeze(0)
        T_tm1 = self._standardize_grid_input(T_tm1, 1).unsqueeze(0)
        wept_tm1 = self._standardize_grid_input(wept_tm1, 2).unsqueeze(0)

        p_t = self._standardize_grid_input(p_t, 0).unsqueeze(0)
        T_t = self._standardize_grid_input(T_t, 1).unsqueeze(0)
        wept_t = self._standardize_grid_input(wept_t, 2).unsqueeze(0)

        # Build input: [static, P_{t-1}, T_{t-1}, WEPT_{t-1}, P_t, T_t, WEPT_t]
        x_grid = torch.cat([static, p_tm1, T_tm1, wept_tm1, p_t, T_t, wept_t], dim=0).contiguous()

        # Scalar parameters
        params_t = torch.from_numpy(params).float()
        params_condition = params_t.clone()

        # Compute targets: residuals (delta = t+1 - t)
        if self.predict_residuals:
            # Compute deltas in raw space
            p_t_raw = f["Output/Pressure"][t, ...]
            T_t_raw = f["Output/Temperature"][t, ...]
            wept_t_raw = f["Output/WEPT"][t, ...]

            delta_p = p_tp1 - p_t_raw
            delta_T = T_tp1 - T_t_raw
            delta_wept = wept_tp1 - wept_t_raw

            # Standardize deltas using grid_output_delta stats
            y_grid = torch.stack([
                self._standardize_grid_output(delta_p, 0),
                self._standardize_grid_output(delta_T, 1),
                self._standardize_grid_output(delta_wept, 2),
            ], dim=0)
        else:
            # Predict absolute values (legacy - not recommended)
            y_grid = torch.stack([
                self._standardize_grid_input(p_tp1, 0),
                self._standardize_grid_input(T_tp1, 1),
                self._standardize_grid_input(wept_tp1, 2),
            ], dim=0)

        # Standardize scalar outputs
        y_scalar = torch.stack([
            self._standardize_scalar(scalars_tp1[i], i) for i in range(5)
        ], dim=0)

        # Create validity mask (mask -999 sentinel values)
        valid_mask = torch.from_numpy((p_tp1 != -999) & (T_tp1 != -999)).float()

        # Augmentations (XY only, never Z)
        if self.augment:
            # Add batch dim for rotation
            x_aug = x_grid.unsqueeze(0)
            y_aug = y_grid.unsqueeze(0)
            mask_aug = valid_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, Z, Y, X]

            # Random 90° rotation in XY plane
            k = np.random.randint(0, 4) if self.aug_xy_rot else 0
            if k > 0:
                x_aug = torch.rot90(x_aug, k, dims=(-2, -1))
                y_aug = torch.rot90(y_aug, k, dims=(-2, -1))
                mask_aug = torch.rot90(mask_aug, k, dims=(-2, -1))

            # Random flips in X and Y
            if self.aug_flip and (np.random.rand() < 0.5):
                x_aug = torch.flip(x_aug, dims=[-1])  # flip X
                y_aug = torch.flip(y_aug, dims=[-1])
                mask_aug = torch.flip(mask_aug, dims=[-1])
            if self.aug_flip and (np.random.rand() < 0.5):
                x_aug = torch.flip(x_aug, dims=[-2])  # flip Y
                y_aug = torch.flip(y_aug, dims=[-2])
                mask_aug = torch.flip(mask_aug, dims=[-2])

            x_grid = x_aug.squeeze(0)
            y_grid = y_aug.squeeze(0)
            valid_mask = mask_aug.squeeze(0).squeeze(0)

            # Gaussian noise on inputs
            if self.noise_std > 0:
                noisy = x_grid.clone()
                for i in range(x_grid.shape[0]):
                    if i < len(self.STATIC_KEYS) and i in self.noise_exclude_static_idx:
                        continue
                    noisy[i] = noisy[i] + torch.randn_like(noisy[i]) * self.noise_std
                x_grid = noisy

        return {
            "x_grid": x_grid.float(),  # [C_in, Z, Y, X] - 14 channels
            "params": params_condition.float(),  # [26]
            "y_grid": y_grid.float(),  # [3, Z, Y, X] - P, T, WEPT deltas
            "y_scalar": y_scalar.float(),  # [5]
            "valid_mask": valid_mask.float(),  # [Z, Y, X]
        }

    def get_all_file_paths(self):
        """Return the full sorted list of all file paths (same across all ranks)."""
        return self.file_paths

    def get_trajectory(self, file_path=None):
        """
        Get full trajectory data from a single file for scheduled sampling rollouts.

        IMPORTANT for DDP: Pass a deterministic file_path from self.file_paths to ensure
        all ranks use the same file. Returns None if the file is not loaded in cache.

        Returns a dict containing:
            - static: [C_static, Z, Y, X] normalized static features
            - params: [26] scalar parameters
            - grid_all: [T, 3, Z, Y, X] all timesteps (P, T, WEPT), normalized with input stats
            - deltas_all: [T-1, 3, Z, Y, X] all deltas, normalized with output stats
            - valid_mask: [Z, Y, X] validity mask
            - T: number of timesteps

        Returns None if the specified file is not loaded in cache (caller should skip).
        """
        import random

        # If no file_path specified, select from loaded files (not DDP-safe)
        if file_path is None:
            loaded_files = self.h5cache.get_loaded_files()
            if not loaded_files:
                import time
                time.sleep(0.01)
                loaded_files = self.h5cache.get_loaded_files()
                if not loaded_files:
                    return None  # No files loaded yet, skip this step
            file_path = random.choice(loaded_files)

        # Check if file is in cache - if not, return None (caller should skip)
        f = self.h5cache.get(file_path)
        if f is None:
            # File not in cache on this rank - return None so caller can skip
            return None

        # Get number of timesteps
        T = f["Output/Pressure"].shape[0]

        # Load static fields
        static_list = []
        for key in self.STATIC_KEYS:
            arr = f[f"Input/{key}"][...]  # [Z, Y, X]
            static_list.append(arr)
        static = np.stack(static_list, axis=0)  # [C_static, Z, Y, X]
        static = self._standardize_static(static)  # Normalized

        # Load scalar params
        params = torch.from_numpy(f["Input/ParamsScalar"][...]).float()  # [26]

        # Load all grid timesteps and normalize
        P_all_raw = f["Output/Pressure"][...]  # [T, Z, Y, X]
        T_all_raw = f["Output/Temperature"][...]
        WEPT_all_raw = f["Output/WEPT"][...]

        # Normalize inputs with grid_input stats
        P_all = torch.stack([self._standardize_grid_input(P_all_raw[t], 0) for t in range(T)], dim=0)
        T_all = torch.stack([self._standardize_grid_input(T_all_raw[t], 1) for t in range(T)], dim=0)
        WEPT_all = torch.stack([self._standardize_grid_input(WEPT_all_raw[t], 2) for t in range(T)], dim=0)
        grid_all = torch.stack([P_all, T_all, WEPT_all], dim=1)  # [T, 3, Z, Y, X]

        # Compute all deltas (raw) and normalize with output stats
        deltas_P = []
        deltas_T = []
        deltas_WEPT = []
        for t in range(T - 1):
            delta_p = P_all_raw[t + 1] - P_all_raw[t]
            delta_T = T_all_raw[t + 1] - T_all_raw[t]
            delta_wept = WEPT_all_raw[t + 1] - WEPT_all_raw[t]
            deltas_P.append(self._standardize_grid_output(delta_p, 0))
            deltas_T.append(self._standardize_grid_output(delta_T, 1))
            deltas_WEPT.append(self._standardize_grid_output(delta_wept, 2))

        deltas_P = torch.stack(deltas_P, dim=0)  # [T-1, Z, Y, X]
        deltas_T = torch.stack(deltas_T, dim=0)
        deltas_WEPT = torch.stack(deltas_WEPT, dim=0)
        deltas_all = torch.stack([deltas_P, deltas_T, deltas_WEPT], dim=1)  # [T-1, 3, Z, Y, X]

        # Validity mask - use first timestep since later timesteps may have all -999 values
        # (some H5 files have -999 fill values in later timesteps for unfilled simulation regions)
        valid_mask = torch.from_numpy((P_all_raw[0] != -999) & (T_all_raw[0] != -999)).float()

        return {
            "static": static.float(),  # [C_static, Z, Y, X]
            "params": params.float(),  # [26]
            "grid_all": grid_all.float(),  # [T, 3, Z, Y, X] - normalized inputs
            "deltas_all": deltas_all.float(),  # [T-1, 3, Z, Y, X] - normalized target deltas
            "valid_mask": valid_mask.float(),  # [Z, Y, X]
            "T": T,
        }

    def denormalize_delta(self, delta_normalized):
        """
        Convert normalized delta back to raw space.
        delta_normalized: [3, Z, Y, X] or [B, 3, Z, Y, X]
        """
        # delta_raw = delta_normalized * std (output deltas only scaled, no mean subtraction)
        std = self.grid_output_std.view(3, 1, 1, 1)
        if delta_normalized.dim() == 5:
            std = std.unsqueeze(0)
        return delta_normalized * std.to(delta_normalized.device)

    def normalize_for_input(self, grid_raw, channel_idx):
        """
        Normalize raw grid values for use as model input.
        grid_raw: [Z, Y, X] raw values
        channel_idx: 0=P, 1=T, 2=WEPT
        """
        m = self.grid_input_mean[channel_idx]
        s = self.grid_input_std[channel_idx].clamp(min=1e-6)
        normalized = (grid_raw - m) / s
        # Handle -999 sentinels
        normalized[grid_raw == -999] = 0
        return normalized

    def denormalize_from_input(self, grid_normalized, channel_idx):
        """
        Convert normalized input grid back to raw space.
        grid_normalized: [..., Z, Y, X]
        channel_idx: 0=P, 1=T, 2=WEPT
        """
        m = self.grid_input_mean[channel_idx].to(grid_normalized.device)
        s = self.grid_input_std[channel_idx].clamp(min=1e-6).to(grid_normalized.device)
        return grid_normalized * s + m
