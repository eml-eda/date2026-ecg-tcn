from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .constants import LABELS


def aggregate_superclasses(scp_codes: Dict[str, float], scp_map: Dict[str, str]) -> List[str]:
    out = set()
    for key in scp_codes.keys():
        cls = scp_map.get(key)
        if cls is not None and isinstance(cls, str) and cls in LABELS:
            out.add(cls)
    return [label for label in LABELS if label in out]


def build_or_load_signal_cache(
    root: Path,
    metadata: pd.DataFrame,
    sampling_rate: int,
    use_cache: bool = True,
) -> np.ndarray:
    try:
        import wfdb  # type: ignore
    except ImportError as e:
        raise ImportError("wfdb is required to read PTB-XL. Install it with: pip install wfdb") from e

    cache_name = f"signals_sr{sampling_rate}.npy"
    cache_path = root / cache_name

    if use_cache and cache_path.exists():
        cached = np.load(cache_path, mmap_mode="r")
        expected_shape = (len(metadata), 12, 1000 if sampling_rate == 100 else 5000)
        if cached.shape == expected_shape:
            return cached
        print(
            f"WARNING: ignoring stale cache {cache_path.name}: "
            f"expected shape {expected_shape}, found {cached.shape}. Rebuilding cache."
        )

    col = "filename_lr" if sampling_rate == 100 else "filename_hr"
    seq_len = 1000 if sampling_rate == 100 else 5000

    if use_cache:
        arr = np.lib.format.open_memmap(
            cache_path,
            mode="w+",
            dtype=np.float32,
            shape=(len(metadata), 12, seq_len),
        )
    else:
        arr = np.empty((len(metadata), 12, seq_len), dtype=np.float32)

    iterator = tqdm(enumerate(metadata[col].tolist()), total=len(metadata), desc=f"Loading PTB-XL {sampling_rate} Hz")
    for i, rel_path in iterator:
        signal, _ = wfdb.rdsamp(str(root / rel_path))
        arr[i] = signal.T.astype(np.float32)

    if use_cache:
        del arr
        return np.load(cache_path, mmap_mode="r")
    return arr


def filter_ptbxl_records_with_files(
    root: Path,
    metadata: pd.DataFrame,
    sampling_rate: int,
) -> pd.DataFrame:
    col = "filename_lr" if sampling_rate == 100 else "filename_hr"
    keep_mask = np.ones(len(metadata), dtype=bool)
    missing_examples: List[str] = []

    iterator = tqdm(
        enumerate(metadata[col].tolist()),
        total=len(metadata),
        desc=f"Checking PTB-XL {sampling_rate} Hz files",
    )
    for i, rel_path in iterator:
        base_path = root / rel_path
        missing_parts = []
        if not base_path.with_suffix(".hea").exists():
            missing_parts.append(".hea")
        if not base_path.with_suffix(".dat").exists():
            missing_parts.append(".dat")
        if missing_parts:
            keep_mask[i] = False
            if len(missing_examples) < 5:
                missing_examples.append(f"{rel_path} ({', '.join(missing_parts)})")

    if keep_mask.all():
        return metadata

    missing_count = int((~keep_mask).sum())
    print(
        f"WARNING: skipping {missing_count} PTB-XL records with missing files "
        f"for {sampling_rate} Hz."
    )
    if missing_examples:
        print("Missing examples:", "; ".join(missing_examples))
    return metadata.loc[keep_mask].copy()


def load_ptbxl(
    root: Path,
    sampling_rate: int,
    use_cache: bool,
    max_records: int = 0,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    db = pd.read_csv(root / "ptbxl_database.csv", index_col="ecg_id")
    db["scp_codes"] = db["scp_codes"].apply(ast.literal_eval)

    scp = pd.read_csv(root / "scp_statements.csv", index_col=0)
    scp = scp[scp["diagnostic"] == 1]
    scp_map = scp["diagnostic_class"].dropna().to_dict()

    db["superclass"] = db["scp_codes"].apply(lambda d: aggregate_superclasses(d, scp_map))
    db = db[db["superclass"].map(len) > 0].copy()
    db = filter_ptbxl_records_with_files(root=root, metadata=db, sampling_rate=sampling_rate)

    if max_records > 0:
        db = db.iloc[:max_records].copy()

    labels = np.zeros((len(db), len(LABELS)), dtype=np.float32)
    for i, labs in enumerate(db["superclass"].tolist()):
        for lab in labs:
            labels[i, LABELS.index(lab)] = 1.0

    signals = build_or_load_signal_cache(root=root, metadata=db, sampling_rate=sampling_rate, use_cache=use_cache)
    return signals, labels, db


def compute_channel_mean_std(signals: np.ndarray, indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n_channels = signals.shape[1]
    total_count = 0
    sum_x = np.zeros(n_channels, dtype=np.float64)
    sum_x2 = np.zeros(n_channels, dtype=np.float64)

    for idx in tqdm(indices.tolist(), desc="Computing train mean/std"):
        x = np.asarray(signals[idx], dtype=np.float32)
        sum_x += x.sum(axis=1, dtype=np.float64)
        sum_x2 += np.square(x, dtype=np.float64).sum(axis=1, dtype=np.float64)
        total_count += x.shape[1]

    mean = sum_x / max(total_count, 1)
    var = (sum_x2 / max(total_count, 1)) - np.square(mean)
    var = np.maximum(var, 1e-8)
    std = np.sqrt(var)
    return mean.astype(np.float32), std.astype(np.float32)


class PTBXLDataset(Dataset):
    def __init__(
        self,
        signals: np.ndarray,
        labels: np.ndarray,
        indices: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
        input_clip: float = 8.0,
        augment: bool = False,
    ) -> None:
        self.signals = signals
        self.labels = labels
        self.indices = indices.astype(np.int64)
        self.mean = mean.astype(np.float32)[:, None]
        self.std = std.astype(np.float32)[:, None]
        self.input_clip = float(input_clip)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.indices)

    def _augment(self, x: np.ndarray) -> np.ndarray:
        gain = np.random.uniform(0.95, 1.05)
        x = x * gain
        if np.random.rand() < 0.5:
            shift = np.random.randint(-5, 6)
            x = np.roll(x, shift=shift, axis=1)
        if np.random.rand() < 0.5:
            noise = np.random.normal(0.0, 0.005, size=x.shape).astype(np.float32)
            x = x + noise
        return x

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        idx = self.indices[i]
        x = np.asarray(self.signals[idx], dtype=np.float32).copy()
        y = self.labels[idx].astype(np.float32)

        x = (x - self.mean) / self.std
        x = np.clip(x, -self.input_clip, self.input_clip)

        if self.augment:
            x = self._augment(x)

        return torch.from_numpy(x), torch.from_numpy(y)


@dataclass
class SplitData:
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray


def build_splits(metadata: pd.DataFrame) -> SplitData:
    folds = metadata["strat_fold"].to_numpy()
    train_idx = np.where(folds <= 8)[0]
    val_idx = np.where(folds == 9)[0]
    test_idx = np.where(folds == 10)[0]
    return SplitData(train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)


def build_loaders(
    signals: np.ndarray,
    labels: np.ndarray,
    metadata: pd.DataFrame,
    batch_size: int,
    num_workers: int,
    input_clip: float,
) -> Tuple[Dict[str, DataLoader], np.ndarray, np.ndarray]:
    splits = build_splits(metadata)
    mean, std = compute_channel_mean_std(signals, splits.train_idx)

    ds_train = PTBXLDataset(signals, labels, splits.train_idx, mean, std, input_clip=input_clip, augment=True)
    ds_val = PTBXLDataset(signals, labels, splits.val_idx, mean, std, input_clip=input_clip, augment=False)
    ds_test = PTBXLDataset(signals, labels, splits.test_idx, mean, std, input_clip=input_clip, augment=False)

    loaders = {
        "train": DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True),
        "val": DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
        "test": DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
    }
    return loaders, mean, std
