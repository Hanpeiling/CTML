from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from sklearn.model_selection import StratifiedKFold

from config import (
    DEFAULT_RANDOM_STATE,
    GROUP_COL,
    ID_TO_TASK,
    TARGET_LOG_COL,
    TASK_COL,
    TASK_TO_ID,
)
from datasets import MultimodalGraphDataset
from graph_utils import GraphBatch, collate_graph_records
from metrics_ext import regression_metrics, summarize_metric_table
from models import MultiTaskAttentionRegressor
from preprocessing import ContextPreprocessor
from lc50_auxiliary import (
    load_and_curate_lc50_fish,
    pretrain_lc50_backbone,
    load_lc50_pretrained_checkpoint,
    load_lc50_state_into_multitask,
    make_lc50_loader,
    train_lc50_aux_steps,
)


_LC50_FRAME = None
_LC50_PREP = None
_LC50_MODULE_STATES = None
_LC50_BATCH_SIZE = 128
_LC50_JOINT_STEPS = 4
_LC50_NUM_WORKERS = 0

_MEASUREMENT_ENCODING_MODE = "hybrid"
_USE_FINGERPRINT_EXPERT = True
_USE_EVIDENCE_ATTENTION = True
_USE_RECORD_QUALITY_GATE = True
_USE_TASK_RESIDUAL = True


def _progress(message: str) -> None:
    """Timestamped, flushed console progress for long Windows runs."""
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)



def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def choose_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)



@dataclass
class TaskTargetScaler:

    global_mean: float
    global_std: float
    task_stats: Dict[str, Tuple[float, float]]

    @classmethod
    def fit(cls, frame: pd.DataFrame) -> "TaskTargetScaler":
        y = frame[TARGET_LOG_COL].astype(float)
        global_mean = float(y.mean())
        global_std = float(y.std(ddof=0))
        if not np.isfinite(global_std) or global_std < 1e-8:
            global_std = 1.0

        task_stats: Dict[str, Tuple[float, float]] = {}
        for task, sub in frame.groupby(TASK_COL):
            vals = sub[TARGET_LOG_COL].astype(float)
            mean = float(vals.mean())
            std = float(vals.std(ddof=0))
            if not np.isfinite(std) or std < 1e-8:
                std = global_std
            task_stats[str(task)] = (mean, std)
        return cls(global_mean, global_std, task_stats)

    def _stats_for_task(self, task: str) -> Tuple[float, float]:
        return self.task_stats.get(str(task), (self.global_mean, self.global_std))

    def transform_frame(self, frame: pd.DataFrame) -> np.ndarray:
        out = np.empty(len(frame), dtype=np.float32)
        for i, (_, row) in enumerate(frame.iterrows()):
            mean, std = self._stats_for_task(str(row[TASK_COL]))
            out[i] = (float(row[TARGET_LOG_COL]) - mean) / std
        return out

    def inverse_arrays(
        self,
        scaled_values: np.ndarray,
        task_ids: np.ndarray,
    ) -> np.ndarray:
        scaled_values = np.asarray(scaled_values, dtype=float).reshape(-1)
        task_ids = np.asarray(task_ids, dtype=int).reshape(-1)
        out = np.empty_like(scaled_values, dtype=float)
        for i, (value, task_id) in enumerate(zip(scaled_values, task_ids)):
            task = ID_TO_TASK[int(task_id)]
            mean, std = self._stats_for_task(task)
            out[i] = value * std + mean
        return out

    def to_dict(self) -> dict:
        return {
            "global_mean": self.global_mean,
            "global_std": self.global_std,
            "task_stats": {
                task: {"mean": mean, "std": std}
                for task, (mean, std) in self.task_stats.items()
            },
        }



TRANSFER_FEATURE_DIM = 5
TRANSFER_FEATURE_NAMES = (
    "mean_z", "median_z", "sd_z", "normalized_log_count", "observed_mask"
)

RECORD_TRANSFER_FEATURE_DIM = 9
RECORD_TRANSFER_FEATURE_NAMES = (
    "source_outcome_z",
    "similarity_tk",
    "similarity_species",
    "similarity_environment",
    "similarity_endpoint",
    "similarity_measurement",
    "combined_context_similarity",
    "source_count_strength",
    "observed_mask",
)


@dataclass
class CrossTaskEvidenceStore:

    table: Dict[Tuple[str, int], np.ndarray]
    max_count: int
    n_tasks: int = 8

    @classmethod
    def fit(
        cls,
        frame: pd.DataFrame,
        scaler: TaskTargetScaler,
        n_tasks: int = 8,
    ) -> "CrossTaskEvidenceStore":
        work = frame[[GROUP_COL, TASK_COL, TARGET_LOG_COL]].copy()
        z = np.empty(len(work), dtype=float)
        for i, (_, row) in enumerate(work.iterrows()):
            mean, std = scaler._stats_for_task(str(row[TASK_COL]))
            z[i] = (float(row[TARGET_LOG_COL]) - mean) / max(float(std), 1e-8)
        work["_evidence_z"] = z

        grouped = (
            work.groupby([GROUP_COL, TASK_COL])["_evidence_z"]
            .agg(["mean", "median", "std", "count"])
            .reset_index()
        )
        grouped["std"] = grouped["std"].fillna(0.0)
        max_count = max(int(grouped["count"].max()) if len(grouped) else 1, 1)
        denom = math.log1p(max_count)

        table: Dict[Tuple[str, int], np.ndarray] = {}
        for _, row in grouped.iterrows():
            task_name = str(row[TASK_COL]).lower()
            if task_name not in TASK_TO_ID:
                continue
            task_id = int(TASK_TO_ID[task_name])
            count_feature = math.log1p(float(row["count"])) / max(denom, 1e-8)
            table[(str(row[GROUP_COL]), task_id)] = np.asarray(
                [
                    float(row["mean"]),
                    float(row["median"]),
                    float(row["std"]),
                    float(count_feature),
                    1.0,
                ],
                dtype=np.float32,
            )
        return cls(table=table, max_count=max_count, n_tasks=int(n_tasks))

    def transform(self, frame: pd.DataFrame, enabled: bool = True) -> np.ndarray:
        out = np.zeros((len(frame), self.n_tasks, TRANSFER_FEATURE_DIM), dtype=np.float32)
        if not enabled:
            return out
        for i, (_, row) in enumerate(frame.iterrows()):
            parent = str(row[GROUP_COL])
            target_id = int(TASK_TO_ID[str(row[TASK_COL]).lower()])
            for source_id in range(self.n_tasks):
                if source_id == target_id:
                    continue
                value = self.table.get((parent, source_id))
                if value is not None:
                    out[i, source_id, :] = value
        return out

    def support_summary(self, frame: pd.DataFrame) -> pd.DataFrame:
        arr = self.transform(frame, enabled=True)
        mask = arr[..., -1] > 0.5
        rows = []
        for i, (_, row) in enumerate(frame.iterrows()):
            rows.append({
                "row_index": int(row.name),
                "task": str(row[TASK_COL]),
                "parent": str(row[GROUP_COL]),
                "n_source_tasks": int(mask[i].sum()),
                "mean_source_count_strength": float(arr[i, :, 3][mask[i]].mean()) if mask[i].any() else 0.0,
            })
        return pd.DataFrame(rows)

    def auxiliary_targets(
        self,
        frame: pd.DataFrame,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
      
        targets = np.zeros((len(frame), self.n_tasks), dtype=np.float32)
        masks = np.zeros((len(frame), self.n_tasks), dtype=np.float32)
        parent_counts = frame.groupby(GROUP_COL).size().astype(float).to_dict()
        parent_weights = np.ones((len(frame), 1), dtype=np.float32)
        for i, (_, row) in enumerate(frame.iterrows()):
            parent = str(row[GROUP_COL])
            parent_weights[i, 0] = 1.0 / max(float(parent_counts.get(row[GROUP_COL], 1.0)), 1.0)
            for task_id in range(self.n_tasks):
                value = self.table.get((parent, task_id))
                if value is not None and float(value[-1]) > 0.5:
                    targets[i, task_id] = float(value[0])  # training-only mean_z
                    masks[i, task_id] = 1.0
        return targets, masks, parent_weights


@dataclass
class ContextAwareRecordEvidenceStore:

    records: Dict[Tuple[str, int], Dict[str, np.ndarray]]
    max_count: int
    per_source_task: int = 4
    n_tasks: int = 8

    @staticmethod
    def _similarity(source: np.ndarray, target: np.ndarray) -> np.ndarray:
        source = np.asarray(source, dtype=np.float32)
        target = np.asarray(target, dtype=np.float32).reshape(1, -1)
        if source.ndim != 2 or source.shape[1] == 0:
            return np.ones((source.shape[0],), dtype=np.float32)
        distance = np.mean(np.abs(source - target), axis=1)
        return (1.0 / (1.0 + distance)).astype(np.float32)

    @staticmethod
    def _cosine_similarity(source: np.ndarray, target: np.ndarray) -> np.ndarray:
        source = np.asarray(source, dtype=np.float32)
        target = np.asarray(target, dtype=np.float32).reshape(-1)
        if source.ndim != 2 or source.shape[1] == 0:
            return np.ones((source.shape[0],), dtype=np.float32)
        tnorm = float(np.linalg.norm(target))
        snorm = np.linalg.norm(source, axis=1)
        denom = np.clip(snorm * max(tnorm, 1e-8), 1e-8, None)
        sim = np.sum(source * target.reshape(1, -1), axis=1) / denom
        return np.clip(sim, 0.0, 1.0).astype(np.float32)

    @classmethod
    def fit(
        cls,
        frame: pd.DataFrame,
        preprocessor: ContextPreprocessor,
        scaler: TaskTargetScaler,
        per_source_task: int = 4,
        n_tasks: int = 8,
    ) -> "ContextAwareRecordEvidenceStore":
        context = preprocessor.transform(frame)
        z = scaler.transform_frame(frame).astype(np.float32)
        records: Dict[Tuple[str, int], Dict[str, np.ndarray]] = {}

        positions: Dict[Tuple[str, int], List[int]] = {}
        for pos, (_, row) in enumerate(frame.iterrows()):
            task_id = int(TASK_TO_ID[str(row[TASK_COL]).lower()])
            key = (str(row[GROUP_COL]), task_id)
            positions.setdefault(key, []).append(pos)

        max_count = 1
        for key, ids in positions.items():
            idx = np.asarray(ids, dtype=int)
            max_count = max(max_count, int(len(idx)))
            records[key] = {
                "z": z[idx],
                "tk": np.asarray(context["tk"][idx], dtype=np.float32),
                "species": np.asarray(context["species"][idx], dtype=np.float32),
                "environment": np.asarray(context["environment"][idx], dtype=np.float32),
                "endpoint": np.asarray(context["endpoint"][idx], dtype=np.float32),
                "measurement": np.asarray(context["measurement"][idx], dtype=np.float32),
            }
        return cls(
            records=records,
            max_count=max_count,
            per_source_task=max(1, int(per_source_task)),
            n_tasks=int(n_tasks),
        )

    @property
    def max_records(self) -> int:
        return max(1, (self.n_tasks - 1) * self.per_source_task)

    def transform(
        self,
        frame: pd.DataFrame,
        target_context: Dict[str, np.ndarray],
        enabled: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        features = np.zeros(
            (len(frame), self.max_records, RECORD_TRANSFER_FEATURE_DIM),
            dtype=np.float32,
        )
        task_ids = np.full((len(frame), self.max_records), -1.0, dtype=np.float32)
        if not enabled:
            return features, task_ids

        log_denom = max(math.log1p(self.max_count), 1e-8)
        for i, (_, row) in enumerate(frame.iterrows()):
            parent = str(row[GROUP_COL])
            target_id = int(TASK_TO_ID[str(row[TASK_COL]).lower()])
            slot = 0
            for source_id in range(self.n_tasks):
                if source_id == target_id:
                    continue
                rec = self.records.get((parent, source_id))
                if rec is None:
                    continue

                sim_tk = self._similarity(rec["tk"], target_context["tk"][i])
                sim_species = self._similarity(rec["species"], target_context["species"][i])
                sim_environment = self._similarity(
                    rec["environment"], target_context["environment"][i]
                )
                sim_endpoint = self._similarity(rec["endpoint"], target_context["endpoint"][i])
                sim_measurement = self._cosine_similarity(
                    rec["measurement"], target_context["measurement"][i]
                )

                combined = (
                    0.15 * sim_tk
                    + 0.20 * sim_species
                    + 0.15 * sim_environment
                    + 0.25 * sim_endpoint
                    + 0.25 * sim_measurement
                )
                order = np.argsort(-combined, kind="mergesort")[: self.per_source_task]
                count_strength = math.log1p(len(rec["z"])) / log_denom
                for j in order:
                    if slot >= self.max_records:
                        break
                    features[i, slot, :] = np.asarray(
                        [
                            float(rec["z"][j]),
                            float(sim_tk[j]),
                            float(sim_species[j]),
                            float(sim_environment[j]),
                            float(sim_endpoint[j]),
                            float(sim_measurement[j]),
                            float(combined[j]),
                            float(count_strength),
                            1.0,
                        ],
                        dtype=np.float32,
                    )
                    task_ids[i, slot] = float(source_id)
                    slot += 1
        return features, task_ids

    def support_summary(
        self,
        frame: pd.DataFrame,
        target_context: Dict[str, np.ndarray],
    ) -> pd.DataFrame:
        features, task_ids = self.transform(frame, target_context, enabled=True)
        valid = features[..., -1] > 0.5
        rows = []
        for i, (_, row) in enumerate(frame.iterrows()):
            ids = task_ids[i][valid[i]].astype(int)
            rows.append({
                "row_index": int(row.name),
                "task": str(row[TASK_COL]),
                "parent": str(row[GROUP_COL]),
                "n_record_evidence": int(valid[i].sum()),
                "n_source_tasks_with_records": int(len(set(ids.tolist()))) if len(ids) else 0,
                "mean_combined_context_similarity": float(
                    features[i, valid[i], 6].mean()
                ) if valid[i].any() else 0.0,
                "max_combined_context_similarity": float(
                    features[i, valid[i], 6].max()
                ) if valid[i].any() else 0.0,
            })
        return pd.DataFrame(rows)


class TaskBalancedBatchSampler:

    def __init__(
        self,
        frame: pd.DataFrame,
        batch_size: int,
        power: float = 0.75,
        seed: int = 42,
    ):
        self.batch_size = int(batch_size)
        self.power = float(power)
        self.seed = int(seed)
        self.epoch = 0
        labels = np.asarray(
            [TASK_TO_ID[str(x).lower()] for x in frame[TASK_COL].astype(str)],
            dtype=int,
        )
        self.task_indices = {
            int(t): np.flatnonzero(labels == int(t)).astype(int)
            for t in sorted(np.unique(labels))
        }
        self.active_tasks = list(self.task_indices)
        if self.batch_size < len(self.active_tasks):
            raise ValueError(
                f"batch_size={self.batch_size} is smaller than active tasks={len(self.active_tasks)}"
            )
        self.n_records = len(frame)
        self.n_batches = max(1, int(math.ceil(self.n_records / self.batch_size)))
        self.quotas = self._compute_quotas()

    def _compute_quotas(self) -> Dict[int, int]:
        counts = np.asarray([len(self.task_indices[t]) for t in self.active_tasks], dtype=float)
        probs = np.power(np.maximum(counts, 1.0), self.power)
        probs = probs / probs.sum()
        remaining = self.batch_size - len(self.active_tasks)
        raw_extra = probs * remaining
        extra = np.floor(raw_extra).astype(int)
        left = remaining - int(extra.sum())
        if left > 0:
            order = np.argsort(-(raw_extra - extra))
            for j in order[:left]:
                extra[j] += 1
        return {t: int(1 + extra[i]) for i, t in enumerate(self.active_tasks)}

    def __len__(self) -> int:
        return self.n_batches

    def __iter__(self):
        rng = np.random.default_rng(self.seed + 1009 * self.epoch)
        self.epoch += 1
        pools = {t: rng.permutation(idx).tolist() for t, idx in self.task_indices.items()}
        cursors = {t: 0 for t in self.active_tasks}

        def draw(task: int, n: int) -> List[int]:
            picked: List[int] = []
            while len(picked) < n:
                pool = pools[task]
                cur = cursors[task]
                available = len(pool) - cur
                take = min(n - len(picked), available)
                if take > 0:
                    picked.extend(pool[cur:cur + take])
                    cursors[task] += take
                if len(picked) < n:
                    pools[task] = rng.permutation(self.task_indices[task]).tolist()
                    cursors[task] = 0
            return picked

        for _ in range(self.n_batches):
            batch: List[int] = []
            for task in self.active_tasks:
                batch.extend(draw(task, self.quotas[task]))
            rng.shuffle(batch)
            yield batch


def task_balanced_group_kfold_split(
    frame: pd.DataFrame,
    n_splits: int,
    random_state: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:

    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")

    work = frame.reset_index(drop=True).copy()
    task_values = sorted(work[TASK_COL].dropna().astype(str).unique())
    if not task_values:
        raise ValueError("No task labels available for cross-validation.")

    parent_counts = [
        work.loc[work[TASK_COL].astype(str) == task, GROUP_COL].nunique()
        for task in task_values
    ]
    parent_counts = [int(x) for x in parent_counts if int(x) >= 2]
    if not parent_counts:
        raise ValueError("Too few task-specific parent structures for cross-validation.")
    n_splits = min(int(n_splits), min(parent_counts))
    if n_splits < 2:
        raise ValueError("Too few task-specific parent structures for cross-validation.")

    fold_for_row = np.full(len(work), -1, dtype=int)

    for task_i, task in enumerate(task_values):
        mask = work[TASK_COL].astype(str) == task
        sub = work.loc[mask].copy()
        groups = sub[GROUP_COL].dropna().astype(str).unique().tolist()
        if not groups:
            continue

        
        group_counts = sub.groupby(GROUP_COL).size().reindex(groups, fill_value=0).astype(float)

        
        species_table = (
            sub.assign(_species=sub["Species"].astype("string").fillna("__MISSING__"))
            .groupby([GROUP_COL, "_species"])
            .size()
            .unstack(fill_value=0)
            .reindex(index=groups, fill_value=0)
        )
        species_counts = species_table.to_numpy(dtype=float)
        species_totals = species_counts.sum(axis=0)
        totals = group_counts.to_numpy(dtype=float)

        if species_counts.shape[1]:
            rarity = 1.0 / np.maximum(species_totals, 1.0)
            rare_priority = (species_counts > 0).dot(
                rarity * max(float(totals.mean()), 1.0)
            )
        else:
            rare_priority = np.zeros(len(groups), dtype=float)

        rng = np.random.default_rng(random_state + 1009 * (task_i + 1))
        priority = totals + 1.5 * rare_priority + rng.random(len(groups)) * 1e-6
        order = np.argsort(-priority)

        target_rows = totals.sum() / n_splits
        target_groups = len(groups) / n_splits
        target_species = species_totals / n_splits if len(species_totals) else species_totals

        fold_rows = np.zeros(n_splits, dtype=float)
        fold_groups = np.zeros(n_splits, dtype=float)
        fold_species = np.zeros((n_splits, len(species_totals)), dtype=float)
        assignment: Dict[str, int] = {}

        for pos in order:
            g = str(groups[pos])
            g_rows = totals[pos]
            g_species = species_counts[pos] if species_counts.shape[1] else np.zeros(0)
            candidates = list(range(n_splits))
            rng.shuffle(candidates)
            best_fold = None
            best_score = None
            for fold in candidates:
                fr = fold_rows.copy(); fg = fold_groups.copy(); fs = fold_species.copy()
                fr[fold] += g_rows
                fg[fold] += 1
                if fs.shape[1]:
                    fs[fold] += g_species

                row_score = np.mean(((fr - target_rows) / max(target_rows, 1.0)) ** 2)
                group_score = np.mean(((fg - target_groups) / max(target_groups, 1.0)) ** 2)
                if fs.shape[1]:
                    species_scale = np.maximum(target_species, 1.0)
                    species_score = np.mean(((fs - target_species) / species_scale) ** 2)
                else:
                    species_score = 0.0
                score = 4.0 * species_score + 1.5 * row_score + 0.5 * group_score
                if best_score is None or score < best_score:
                    best_score = score
                    best_fold = fold

            assignment[g] = int(best_fold)
            fold_rows[best_fold] += g_rows
            fold_groups[best_fold] += 1
            if fold_species.shape[1]:
                fold_species[best_fold] += g_species

        task_positions = np.flatnonzero(mask.to_numpy())
        task_groups = work.loc[mask, GROUP_COL].astype(str).to_numpy()
        for pos, g in zip(task_positions, task_groups):
            fold_for_row[pos] = assignment[str(g)]

    if np.any(fold_for_row < 0):
        raise RuntimeError("Some rows were not assigned to a task-specific CV fold.")

    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    for fold in range(n_splits):
        valid_idx = np.flatnonzero(fold_for_row == fold)
        train_idx = np.flatnonzero(fold_for_row != fold)
        if len(valid_idx) == 0 or len(train_idx) == 0:
            continue

        train_part = work.iloc[train_idx]
        valid_part = work.iloc[valid_idx]
        for task in task_values:
            v = set(valid_part.loc[valid_part[TASK_COL].astype(str) == task, GROUP_COL].astype(str))
            t = set(train_part.loc[train_part[TASK_COL].astype(str) == task, GROUP_COL].astype(str))
            if not v.isdisjoint(t):
                raise AssertionError(f"Task-specific CV leakage for task={task}, fold={fold}")
        splits.append((train_idx, valid_idx))

    return splits

def _make_loader(
    frame: pd.DataFrame,
    preprocessor: ContextPreprocessor,
    scaler: TaskTargetScaler,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    evidence_store: Optional[CrossTaskEvidenceStore] = None,
    record_evidence_store: Optional[ContextAwareRecordEvidenceStore] = None,
    use_cross_task_evidence: bool = True,
    use_record_level_evidence: bool = True,
    record_evidence_per_source: int = 4,
    task_sampling_power: float = 0.75,
    seed: int = 42,
    include_auxiliary_targets: bool = False,
) -> DataLoader:
    contexts = preprocessor.transform(frame)
    if evidence_store is None:
        contexts["transfer"] = np.zeros(
            (len(frame), len(TASK_TO_ID), TRANSFER_FEATURE_DIM), dtype=np.float32
        )
    else:
        contexts["transfer"] = evidence_store.transform(
            frame, enabled=use_cross_task_evidence
        )

   
    if record_evidence_store is None:
        max_records = max(1, (len(TASK_TO_ID) - 1) * int(record_evidence_per_source))
        contexts["transfer_record"] = np.zeros(
            (len(frame), max_records, RECORD_TRANSFER_FEATURE_DIM), dtype=np.float32
        )
        contexts["transfer_record_task_id"] = np.full(
            (len(frame), max_records), -1.0, dtype=np.float32
        )
    else:
        record_features, record_task_ids = record_evidence_store.transform(
            frame, contexts,
            enabled=bool(use_cross_task_evidence and use_record_level_evidence),
        )
        contexts["transfer_record"] = record_features
        contexts["transfer_record_task_id"] = record_task_ids

    if include_auxiliary_targets and evidence_store is not None:
        aux_target, aux_mask, aux_parent_weight = evidence_store.auxiliary_targets(frame)
    else:
        aux_target = np.zeros((len(frame), len(TASK_TO_ID)), dtype=np.float32)
        aux_mask = np.zeros((len(frame), len(TASK_TO_ID)), dtype=np.float32)
        aux_parent_weight = np.ones((len(frame), 1), dtype=np.float32)
    contexts["aux_target"] = aux_target
    contexts["aux_mask"] = aux_mask
    contexts["aux_parent_weight"] = aux_parent_weight

    model_frame = frame.copy()
    model_frame[TARGET_LOG_COL] = scaler.transform_frame(frame)
    ds = MultimodalGraphDataset(model_frame, contexts)

    if shuffle:
        batch_sampler = TaskBalancedBatchSampler(
            frame,
            batch_size=int(batch_size),
            power=float(task_sampling_power),
            seed=int(seed),
        )
        return DataLoader(
            ds,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=collate_graph_records,
        )

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_graph_records,
        drop_last=False,
    )


def _task_parent_weight_map(
    frame: pd.DataFrame,
    power: float,
) -> Dict[int, float]:
 
    counts = frame.groupby([TASK_COL, GROUP_COL]).size()
    raw = []
    for idx, row in frame.iterrows():
        n = float(counts.loc[(row[TASK_COL], row[GROUP_COL])])
        raw.append((idx, n ** (-float(power))))
    values = np.array([x[1] for x in raw], dtype=float)
    mean_w = float(values.mean()) if len(values) else 1.0
    if not np.isfinite(mean_w) or mean_w <= 0:
        mean_w = 1.0
    return {int(idx): float(w / mean_w) for idx, w in raw}


def _weighted_mean(losses: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    weights = weights.to(dtype=losses.dtype, device=losses.device)
    return torch.sum(losses * weights) / torch.sum(weights).clamp_min(1e-12)


def task_base_losses(
    pred: torch.Tensor,
    y: torch.Tensor,
    task_id: torch.Tensor,
    sample_weights: torch.Tensor,
    loss_type: str = "mse",
    delta: float = 1.0,
) -> Dict[int, torch.Tensor]:
   
    if str(loss_type).lower() == "huber":
        element = F.huber_loss(pred, y, reduction="none", delta=float(delta))
    elif str(loss_type).lower() == "mse":
        element = (pred - y) ** 2
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    out: Dict[int, torch.Tensor] = {}
    for task in torch.unique(task_id):
        task_int = int(task.detach().cpu())
        mask = task_id == task
        if torch.any(mask):
            out[task_int] = _weighted_mean(element[mask], sample_weights[mask])
    return out


def masked_auxiliary_multitask_loss(
    aux_pred: torch.Tensor,
    aux_target: torch.Tensor,
    aux_mask: torch.Tensor,
    parent_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
   
    if aux_pred.shape != aux_target.shape or aux_pred.shape != aux_mask.shape:
        raise ValueError(
            f"Auxiliary shapes must match: pred={tuple(aux_pred.shape)} "
            f"target={tuple(aux_target.shape)} mask={tuple(aux_mask.shape)}"
        )
    if parent_weights is None:
        parent_weights = torch.ones(
            (aux_pred.shape[0],), dtype=aux_pred.dtype, device=aux_pred.device
        )
    else:
        parent_weights = parent_weights.reshape(-1).to(
            dtype=aux_pred.dtype, device=aux_pred.device
        )

    element = (aux_pred - aux_target) ** 2
    task_losses = []
    for task_id in range(aux_pred.shape[1]):
        valid = aux_mask[:, task_id] > 0.5
        if torch.any(valid):
            task_losses.append(
                _weighted_mean(element[valid, task_id], parent_weights[valid])
            )
    if not task_losses:
        return aux_pred.sum() * 0.0
    return torch.stack(task_losses).mean()


def _normal_or_uncertainty_objectives(
    model: nn.Module,
    base_losses: Dict[int, torch.Tensor],
    use_uncertainty_weighting: bool,
) -> Dict[int, torch.Tensor]:
  
    objectives: Dict[int, torch.Tensor] = {}
    for task_int, base_loss in base_losses.items():
        if use_uncertainty_weighting and hasattr(model, "uncertainty_weighted_objective"):
            objectives[task_int] = model.uncertainty_weighted_objective(task_int, base_loss)
        else:
            objectives[task_int] = base_loss
    return objectives


def _pcgrad_dot(
    grad_a: Sequence[Optional[torch.Tensor]],
    grad_b: Sequence[Optional[torch.Tensor]],
) -> torch.Tensor:
    value = None
    for ga, gb in zip(grad_a, grad_b):
        if ga is None or gb is None:
            continue
        term = torch.sum(ga * gb)
        value = term if value is None else value + term
    if value is None:
        # This should not normally happen for the shared backbone.
        return torch.tensor(0.0)
    return value


def _pcgrad_norm_sq(
    grad: Sequence[Optional[torch.Tensor]],
) -> torch.Tensor:
    value = None
    for g in grad:
        if g is None:
            continue
        term = torch.sum(g * g)
        value = term if value is None else value + term
    if value is None:
        return torch.tensor(0.0)
    return value


def _project_conflicting_gradients(
    task_gradients: List[List[Optional[torch.Tensor]]],
) -> Tuple[List[Optional[torch.Tensor]], Dict[str, float]]:
   
    if not task_gradients:
        return [], {
            "pair_checks": 0.0, "conflicts": 0.0, "conflict_rate": 0.0,
            "projection_events": 0.0, "mean_cosine": np.nan,
            "mean_negative_cosine": np.nan,
        }
    if len(task_gradients) == 1:
        merged = [None if g is None else g.clone() for g in task_gradients[0]]
        return merged, {
            "pair_checks": 0.0, "conflicts": 0.0, "conflict_rate": 0.0,
            "projection_events": 0.0, "mean_cosine": np.nan,
            "mean_negative_cosine": np.nan,
        }

    raw_cosines: List[float] = []
    negative_cosines: List[float] = []
    conflicts = 0
    pair_checks = 0
    for i in range(len(task_gradients)):
        for j in range(i + 1, len(task_gradients)):
            dot = _pcgrad_dot(task_gradients[i], task_gradients[j])
            ni = _pcgrad_norm_sq(task_gradients[i])
            nj = _pcgrad_norm_sq(task_gradients[j])
            denom = torch.sqrt(ni.clamp_min(1e-12) * nj.clamp_min(1e-12))
            cos = float((dot / denom).detach().cpu()) if float(denom.detach().cpu()) > 0 else 0.0
            raw_cosines.append(cos)
            pair_checks += 1
            if cos < 0.0:
                conflicts += 1
                negative_cosines.append(cos)

    projected: List[List[Optional[torch.Tensor]]] = [
        [None if g is None else g.clone() for g in grads]
        for grads in task_gradients
    ]
    projection_events = 0
    order = list(range(len(projected)))
    for i in range(len(projected)):
        others = [j for j in order if j != i]
        random.shuffle(others)
        for j in others:
            dot = _pcgrad_dot(projected[i], task_gradients[j])
            if float(dot.detach().cpu()) >= 0.0:
                continue
            norm_sq = _pcgrad_norm_sq(task_gradients[j])
            if float(norm_sq.detach().cpu()) <= 1e-12:
                continue
            coeff = dot / norm_sq
            changed = False
            for k, (gi, gj) in enumerate(zip(projected[i], task_gradients[j])):
                if gi is not None and gj is not None:
                    projected[i][k] = gi - coeff * gj
                    changed = True
            if changed:
                projection_events += 1

    merged: List[Optional[torch.Tensor]] = []
    for p_idx in range(len(projected[0])):
        available = [g[p_idx] for g in projected if g[p_idx] is not None]
        merged.append(None if not available else torch.stack(available, dim=0).mean(dim=0))

    diagnostics = {
        "pair_checks": float(pair_checks),
        "conflicts": float(conflicts),
        "conflict_rate": float(conflicts / pair_checks) if pair_checks else 0.0,
        "projection_events": float(projection_events),
        "mean_cosine": float(np.mean(raw_cosines)) if raw_cosines else np.nan,
        "mean_negative_cosine": float(np.mean(negative_cosines)) if negative_cosines else np.nan,
    }
    return merged, diagnostics

def _make_optimizer(
    model: nn.Module,
    lr: float,
    weight_decay: float,
    uncertainty_lr_multiplier: float = 10.0,
) -> torch.optim.Optimizer:
  
    uncertainty_params = []
    other_params = []
    uncertainty_ids = set()
    if hasattr(model, "task_log_vars"):
        uncertainty_params = [model.task_log_vars]
        uncertainty_ids = {id(model.task_log_vars)}

    for param in model.parameters():
        if param.requires_grad and id(param) not in uncertainty_ids:
            other_params.append(param)

    groups = [{
        "params": other_params,
        "weight_decay": float(weight_decay),
        "lr": float(lr),
    }]
    if uncertainty_params:
        groups.append({
            "params": uncertainty_params,
            "weight_decay": 0.0,
            "lr": float(lr) * float(uncertainty_lr_multiplier),
        })
    return AdamW(groups)

def within_task_pairwise_rank_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    task_id: torch.Tensor,
    min_target_delta: float = 0.20,
) -> torch.Tensor:
    
    parts: List[torch.Tensor] = []
    for t in torch.unique(task_id):
        mask = task_id == t
        p = pred[mask]
        y = target[mask]
        if p.numel() < 2:
            continue
        dy = y[:, None] - y[None, :]
        dp = p[:, None] - p[None, :]
        upper = torch.triu(torch.ones_like(dy, dtype=torch.bool), diagonal=1)
        valid = upper & (torch.abs(dy) >= float(min_target_delta))
        if torch.any(valid):
            direction = torch.sign(dy[valid])
            parts.append(F.softplus(-direction * dp[valid]).mean())
    if not parts:
        return pred.sum() * 0.0
    return torch.stack(parts).mean()


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    weight_map: Dict[int, float],
    loss_type: str = "mse",
    huber_delta: float = 1.0,
    grad_clip: float = 5.0,
    use_uncertainty_weighting: bool = True,
    use_pcgrad: bool = False,
    epoch_number: int = 1,
    uncertainty_warmup_epochs: int = 5,
    auxiliary_loss_weight: float = 0.10,
    rank_loss_weight: float = 0.0,
    diagnostics_out: Optional[Dict[str, float]] = None,
) -> float:
    """One epoch of record-level + masked molecular multi-output learning."""
    model.train()
    losses: List[float] = []
    aux_losses: List[float] = []
    rank_losses: List[float] = []
    agg_pairs = agg_conflicts = agg_projections = 0.0
    cosines: List[float] = []
    negative_cosines: List[float] = []
    pcgrad_batches = 0

    uncertainty_active = bool(
        use_uncertainty_weighting and int(epoch_number) > int(uncertainty_warmup_epochs)
    )
    aux_weight = max(float(auxiliary_loss_weight), 0.0)
    rank_weight = max(float(rank_loss_weight), 0.0)

    for batch in loader:
        batch = batch.to(device)
        selected, _all, aux_pred = model(batch)
        row_idx = batch.row_index.detach().cpu().numpy().reshape(-1)
        weights = torch.tensor(
            [weight_map.get(int(i), 1.0) for i in row_idx],
            dtype=selected.dtype,
            device=device,
        )

        base_losses = task_base_losses(
            selected, batch.y, batch.task_id, weights,
            loss_type=loss_type, delta=huber_delta,
        )
        objectives = _normal_or_uncertainty_objectives(
            model, base_losses, use_uncertainty_weighting=uncertainty_active
        )
        if not objectives:
            continue

        objective_values = list(objectives.values())
        main_objective = torch.stack(objective_values).mean()

        aux_loss = selected.sum() * 0.0
        aux_mask = batch.context.get("aux_mask")
        if aux_weight > 0.0 and aux_mask is not None and torch.any(aux_mask > 0.5):
            aux_loss = masked_auxiliary_multitask_loss(
                aux_pred,
                batch.context["aux_target"],
                aux_mask,
                batch.context.get("aux_parent_weight"),
            )
        rank_loss = within_task_pairwise_rank_loss(selected, batch.y, batch.task_id)
        total_objective = main_objective + aux_weight * aux_loss + rank_weight * rank_loss
        optimizer.zero_grad(set_to_none=True)

        if use_pcgrad and len(objective_values) >= 2 and hasattr(model, "pcgrad_shared_parameters"):
            shared_params = list(model.pcgrad_shared_parameters())
            task_grads: List[List[Optional[torch.Tensor]]] = []
            for task_objective in objective_values:
                grads = torch.autograd.grad(
                    task_objective, shared_params, retain_graph=True,
                    create_graph=False, allow_unused=True
                )
                task_grads.append([None if g is None else g.detach().clone() for g in grads])

            aux_shared_grads = [None] * len(shared_params)
            if aux_weight > 0.0 and torch.any(aux_mask > 0.5):
                ag = torch.autograd.grad(
                    aux_weight * aux_loss, shared_params, retain_graph=True,
                    create_graph=False, allow_unused=True
                )
                aux_shared_grads = [None if g is None else g.detach().clone() for g in ag]

            
            total_objective.backward()
            merged_shared, diag = _project_conflicting_gradients(task_grads)
            for param, main_grad, aux_grad in zip(shared_params, merged_shared, aux_shared_grads):
                combined = None
                if main_grad is not None:
                    combined = main_grad
                if aux_grad is not None:
                    combined = aux_grad if combined is None else combined + aux_grad
                if combined is not None:
                    param.grad = combined

            pcgrad_batches += 1
            agg_pairs += diag["pair_checks"]
            agg_conflicts += diag["conflicts"]
            agg_projections += diag["projection_events"]
            if np.isfinite(diag["mean_cosine"]):
                cosines.append(float(diag["mean_cosine"]))
            if np.isfinite(diag["mean_negative_cosine"]):
                negative_cosines.append(float(diag["mean_negative_cosine"]))
        else:
            total_objective.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        if hasattr(model, "task_log_vars"):
            with torch.no_grad():
                model.task_log_vars.clamp_(-4.0, 4.0)

        losses.append(float(total_objective.detach().cpu()))
        aux_losses.append(float(aux_loss.detach().cpu()))

    if diagnostics_out is not None:
        diagnostics_out.clear()
        diagnostics_out.update({
            "pcgrad_batches": float(pcgrad_batches),
            "pcgrad_pair_checks": float(agg_pairs),
            "pcgrad_conflicts": float(agg_conflicts),
            "pcgrad_conflict_rate": float(agg_conflicts / agg_pairs) if agg_pairs else 0.0,
            "pcgrad_projection_events": float(agg_projections),
            "pcgrad_mean_cosine": float(np.mean(cosines)) if cosines else np.nan,
            "pcgrad_mean_negative_cosine": float(np.mean(negative_cosines)) if negative_cosines else np.nan,
            "uncertainty_weighting_active": float(uncertainty_active),
            "auxiliary_loss": float(np.mean(aux_losses)) if aux_losses else np.nan,
            "auxiliary_loss_weight": float(aux_weight),
            "rank_loss": float(np.mean(rank_losses)) if rank_losses else np.nan,
            "rank_loss_weight": float(rank_weight),
        })

    return float(np.mean(losses)) if losses else np.nan

def _uncertainty_history_fields(model: nn.Module) -> Dict[str, float]:
    """Flatten learned task uncertainty weights for CSV training histories."""
    if not hasattr(model, "task_uncertainty_table"):
        return {}
    table = model.task_uncertainty_table()
    fields: Dict[str, float] = {}
    for task_id, task_name in ID_TO_TASK.items():
        stats = table.get(int(task_id), {})
        safe = str(task_name).replace(" ", "_")
        if stats:
            fields[f"task_weight_{safe}"] = float(stats["relative_weight"])
            fields[f"task_logvar_{safe}"] = float(stats["log_variance"])
            if "residual_scale" in stats:
                fields[f"task_residual_scale_{safe}"] = float(stats["residual_scale"])
            if "task_atom_scale" in stats:
                fields[f"task_atom_scale_{safe}"] = float(stats["task_atom_scale"])
            if "record_evidence_scale" in stats:
                fields[f"task_record_evidence_scale_{safe}"] = float(stats["record_evidence_scale"])
    return fields



@torch.no_grad()
def predict_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    scaler: TaskTargetScaler,
    return_attention: bool = False,
) -> pd.DataFrame:
    model.eval()
    rows = []
    source_task_names = [ID_TO_TASK[i] for i in range(len(ID_TO_TASK))]
    for batch in loader:
        batch = batch.to(device)
        if return_attention:
            pred, _all, _aux, attn = model(batch, return_attention=True)
        else:
            pred, _all, _aux = model(batch)
            attn = None

        pred_scaled = pred.detach().cpu().numpy().reshape(-1)
        y_scaled = batch.y.detach().cpu().numpy().reshape(-1)
        task_ids = batch.task_id.detach().cpu().numpy().reshape(-1)
        row_idx = batch.row_index.detach().cpu().numpy().reshape(-1)

        pred_log = scaler.inverse_arrays(pred_scaled, task_ids)
        true_log = scaler.inverse_arrays(y_scaled, task_ids)

        modality_weights = None
        transfer_weights = None
        record_transfer_weights = None
        record_transfer_max_weight = None
        record_transfer_entropy = None
        record_transfer_n_records = None
        residual_scale = None
        task_atom_scale = None
        record_evidence_scale = None
        record_similarity_prior_strength = None
        record_attention_temperature = None
        fingerprint_scale = None
        lc50_expert_scale = None
        if attn is not None:
            if "modality" in attn:
                modality_weights = attn["modality"].detach().cpu().numpy()
            if "transfer_source" in attn:
                transfer_weights = attn["transfer_source"].detach().cpu().numpy()
            if "record_transfer_source" in attn:
                record_transfer_weights = attn["record_transfer_source"].detach().cpu().numpy()
            if "record_transfer_max_weight" in attn:
                record_transfer_max_weight = attn["record_transfer_max_weight"].detach().cpu().numpy()
            if "record_transfer_entropy" in attn:
                record_transfer_entropy = attn["record_transfer_entropy"].detach().cpu().numpy()
            if "record_transfer_n_records" in attn:
                record_transfer_n_records = attn["record_transfer_n_records"].detach().cpu().numpy()
            if "residual_scale" in attn:
                residual_scale = attn["residual_scale"].detach().cpu().numpy()
            if "task_atom_scale" in attn:
                task_atom_scale = attn["task_atom_scale"].detach().cpu().numpy()
            if "record_evidence_scale" in attn:
                record_evidence_scale = attn["record_evidence_scale"].detach().cpu().numpy()
            if "record_similarity_prior_strength" in attn:
                record_similarity_prior_strength = attn["record_similarity_prior_strength"].detach().cpu().numpy()
            if "record_attention_temperature" in attn:
                record_attention_temperature = attn["record_attention_temperature"].detach().cpu().numpy()
            if "fingerprint_scale" in attn:
                fingerprint_scale = attn["fingerprint_scale"].detach().cpu().numpy()
            if "lc50_expert_scale" in attn:
                lc50_expert_scale = attn["lc50_expert_scale"].detach().cpu().numpy()

        transfer = batch.context.get("transfer")
        transfer_np = transfer.detach().cpu().numpy() if transfer is not None else None

        for i in range(len(pred_log)):
            rec = {
                "row_index": int(row_idx[i]),
                "task_id": int(task_ids[i]),
                "task": ID_TO_TASK[int(task_ids[i])],
                "true_log10": float(true_log[i]),
                "pred_log10": float(pred_log[i]),
                "true_mg_L": float(10.0 ** true_log[i]),
                "pred_mg_L": float(10.0 ** pred_log[i]),
            }
            if transfer_np is not None:
                observed = transfer_np[i, :, -1] > 0.5
                rec["transfer_n_source_tasks"] = int(observed.sum())
                rec["transfer_mean_count_strength"] = (
                    float(transfer_np[i, observed, 3].mean()) if observed.any() else 0.0
                )
            if modality_weights is not None:
                for j, name in enumerate(
                    ["graph", "tk", "species", "environment", "endpoint", "measurement"]
                ):
                    rec[f"attention_{name}"] = float(modality_weights[i, j])
            if transfer_weights is not None:
                for j, name in enumerate(source_task_names):
                    rec[f"transfer_attention_{name}"] = float(transfer_weights[i, j])
            if record_transfer_weights is not None:
                for j, name in enumerate(source_task_names):
                    rec[f"record_transfer_attention_{name}"] = float(record_transfer_weights[i, j])
            if record_transfer_max_weight is not None:
                rec["record_transfer_max_weight"] = float(record_transfer_max_weight[i])
            if record_transfer_entropy is not None:
                rec["record_transfer_entropy"] = float(record_transfer_entropy[i])
            if record_transfer_n_records is not None:
                rec["record_transfer_n_records"] = float(record_transfer_n_records[i])
            if residual_scale is not None:
                rec["target_residual_scale"] = float(
                    residual_scale[i, int(task_ids[i])]
                )
            if task_atom_scale is not None:
                rec["target_atom_attention_scale"] = float(
                    task_atom_scale[i, int(task_ids[i])]
                )
            if record_evidence_scale is not None:
                rec["target_record_evidence_scale"] = float(record_evidence_scale[i])
            if record_similarity_prior_strength is not None:
                rec["record_similarity_prior_strength"] = float(record_similarity_prior_strength[i])
            if record_attention_temperature is not None:
                rec["record_attention_temperature"] = float(record_attention_temperature[i])
            if fingerprint_scale is not None:
                rec["fingerprint_expert_scale"] = float(fingerprint_scale[i])
            if lc50_expert_scale is not None:
                rec["lc50_expert_scale"] = float(lc50_expert_scale[i])
            rows.append(rec)
    return pd.DataFrame(rows)


def _macro_task_parent_rmse(
    pred_df: pd.DataFrame,
    frame: pd.DataFrame,
) -> float:
   
    merged = pred_df.copy()
    missing = [c for c in (GROUP_COL, TASK_COL) if c not in merged.columns]
    if missing:
        meta = (
            frame[missing]
            .reset_index()
            .rename(columns={"index": "row_index"})
        )
        merged = merged.merge(meta, on="row_index", how="left")

   
    if TASK_COL not in merged.columns:
        for candidate in (f"{TASK_COL}_x", f"{TASK_COL}_y"):
            if candidate in merged.columns:
                merged[TASK_COL] = merged[candidate]
                break
    if GROUP_COL not in merged.columns:
        for candidate in (f"{GROUP_COL}_x", f"{GROUP_COL}_y"):
            if candidate in merged.columns:
                merged[GROUP_COL] = merged[candidate]
                break

    if TASK_COL not in merged.columns or GROUP_COL not in merged.columns:
        raise KeyError(
            f"Required metadata missing for macro RMSE: {TASK_COL}, {GROUP_COL}. "
            f"Available columns: {list(merged.columns)}"
        )

    task_scores = []
    for _task, task_sub in merged.groupby(TASK_COL):
        parent_scores = []
        for _, parent_sub in task_sub.groupby(GROUP_COL):
            if len(parent_sub):
                parent_scores.append(
                    regression_metrics(
                        parent_sub["true_log10"], parent_sub["pred_log10"]
                    )["RMSE"]
                )
        if parent_scores:
            task_scores.append(float(np.mean(parent_scores)))
    return float(np.mean(task_scores)) if task_scores else float("inf")


def _robust_task_parent_score(
    pred_df: pd.DataFrame,
    frame: pd.DataFrame,
    sd_weight: float = 0.15,
    worst_weight: float = 0.10,
) -> float:
  
    merged = pred_df.copy()
    missing = [c for c in (GROUP_COL, TASK_COL) if c not in merged.columns]
    if missing:
        meta = frame[missing].reset_index().rename(columns={"index": "row_index"})
        merged = merged.merge(meta, on="row_index", how="left")
    task_scores: List[float] = []
    for _task, task_sub in merged.groupby(TASK_COL):
        parent_scores: List[float] = []
        for _, parent_sub in task_sub.groupby(GROUP_COL):
            if len(parent_sub):
                parent_scores.append(regression_metrics(
                    parent_sub["true_log10"], parent_sub["pred_log10"]
                )["RMSE"])
        if parent_scores:
            task_scores.append(float(np.mean(parent_scores)))
    if not task_scores:
        return float("inf")
    arr = np.asarray(task_scores, dtype=float)
    macro = float(np.mean(arr))
    return float(macro + float(sd_weight) * np.std(arr) + float(worst_weight) * (np.max(arr) - macro))


def metric_row(split: str, pred_df: pd.DataFrame, frame: pd.DataFrame) -> dict:
    m = regression_metrics(pred_df["true_log10"], pred_df["pred_log10"])
    return {
        "split": split,
        "n_records": len(frame),
        "n_parent_structures": frame[GROUP_COL].nunique(),
        "n_CAS": frame["Chemicals"].nunique(),
        "n_species": frame["Species"].nunique(),
        "n_indicators": frame["Effect Measurement"].nunique(),
        "n_response_sites": frame["Response Site"].nunique(),
        "n_NOEC": int((frame["Endpoint"] == "NOEC").sum()),
        "n_LOEC": int((frame["Endpoint"] == "LOEC").sum()),
        "TaskParentMacroRMSE": _macro_task_parent_rmse(pred_df, frame),
        **m,
    }



def _build_model(
    context_dims: Dict[str, int],
    params: Dict[str, float | int],
    use_cross_task_evidence: bool = True,
    use_record_level_evidence: bool = True,
    use_task_atom_attention: bool = True,
) -> MultiTaskAttentionRegressor:
    lc50_vocab = (
        int(_LC50_PREP.species_vocab_size)
        if _LC50_PREP is not None else 1
    )
    model = MultiTaskAttentionRegressor(
        context_dims=context_dims,
        hidden_dim=int(params["hidden_dim"]),
        n_layers=int(params["n_layers"]),
        num_heads=int(params.get("num_heads", 1)),
        dropout=float(params["dropout"]),
        n_tasks=len(TASK_TO_ID),
        transfer_feature_dim=TRANSFER_FEATURE_DIM,
        record_transfer_feature_dim=RECORD_TRANSFER_FEATURE_DIM,
        lc50_species_vocab_size=lc50_vocab,
        use_lc50_auxiliary=False,
        use_lc50_expert=bool(_LC50_MODULE_STATES is not None),
        record_attention_temperature=float(params.get("record_attention_temperature", 0.40)),
        use_cross_task_evidence=bool(use_cross_task_evidence),
        use_record_level_evidence=bool(use_record_level_evidence),
        use_task_atom_attention=bool(use_task_atom_attention),
        use_fingerprint_expert=bool(_USE_FINGERPRINT_EXPERT),
        use_evidence_attention=bool(_USE_EVIDENCE_ATTENTION),
        use_record_quality_gate=bool(_USE_RECORD_QUALITY_GATE),
        use_task_residual=bool(_USE_TASK_RESIDUAL),
    )
    if _LC50_MODULE_STATES is not None:
        load_lc50_state_into_multitask(model, _LC50_MODULE_STATES)
    return model


def _make_joint_lc50_loader(seed: int):
    if _LC50_FRAME is None or _LC50_PREP is None or _LC50_MODULE_STATES is None:
        return None
    return make_lc50_loader(
        _LC50_FRAME, _LC50_PREP, batch_size=int(_LC50_BATCH_SIZE),
        shuffle=True, seed=int(seed), num_workers=int(_LC50_NUM_WORKERS),
    )


def _fit_fold(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    params: Dict[str, float | int],
    max_epochs: int,
    patience: int,
    seed: int,
    device: torch.device,
    num_workers: int,
    min_epochs: int = 20,
    smoothing_window: int = 5,
    use_uncertainty_weighting: bool = True,
    use_pcgrad: bool = True,
    uncertainty_lr_multiplier: float = 10.0,
    uncertainty_warmup_epochs: int = 5,
    progress_every: int = 10,
    progress_prefix: str = "",
    use_cross_task_evidence: bool = True,
    use_record_level_evidence: bool = True,
    use_task_atom_attention: bool = True,
    use_auxiliary_loss: bool = True,
) -> Tuple[nn.Module, ContextPreprocessor, TaskTargetScaler, pd.DataFrame, int, pd.DataFrame]:
   
    set_seed(seed)
    prep = ContextPreprocessor(measurement_encoding_mode=_MEASUREMENT_ENCODING_MODE).fit(train_df)
    scaler = TaskTargetScaler.fit(train_df)
    evidence_store = CrossTaskEvidenceStore.fit(train_df, scaler, n_tasks=len(TASK_TO_ID))
    record_evidence_store = ContextAwareRecordEvidenceStore.fit(
        train_df,
        prep,
        scaler,
        per_source_task=int(params.get("record_evidence_per_source", 4)),
        n_tasks=len(TASK_TO_ID),
    )
    weight_map = _task_parent_weight_map(
        train_df, power=float(params.get("chemical_weight_power", 0.125))
    )

    train_loader = _make_loader(
        train_df, prep, scaler, int(params["batch_size"]), True, num_workers,
        evidence_store=evidence_store,
        record_evidence_store=record_evidence_store,
        use_cross_task_evidence=use_cross_task_evidence,
        use_record_level_evidence=use_record_level_evidence,
        record_evidence_per_source=int(params.get("record_evidence_per_source", 4)),
        task_sampling_power=float(params.get("task_sampling_power", 0.75)),
        seed=seed,
        include_auxiliary_targets=bool(use_auxiliary_loss),
    )
    valid_loader = _make_loader(
        valid_df, prep, scaler, int(params["batch_size"]), False, num_workers,
        evidence_store=evidence_store,
        record_evidence_store=record_evidence_store,
        use_cross_task_evidence=use_cross_task_evidence,
        use_record_level_evidence=use_record_level_evidence,
        record_evidence_per_source=int(params.get("record_evidence_per_source", 4)),
        task_sampling_power=float(params.get("task_sampling_power", 0.75)),
        seed=seed,
    )

    model = _build_model(
        prep.output_dims,
        params,
        use_cross_task_evidence=use_cross_task_evidence,
        use_record_level_evidence=use_record_level_evidence,
        use_task_atom_attention=use_task_atom_attention,
    ).to(device)
    optimizer = _make_optimizer(
        model,
        lr=float(params["lr"]),
        weight_decay=float(params["weight_decay"]),
        uncertainty_lr_multiplier=uncertainty_lr_multiplier,
    )
    lc50_loader = None  
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=max(4, int(patience) // 4),
        min_lr=1e-6,
    )

    effective_min_epochs = max(1, min(int(min_epochs), int(max_epochs)))
    best_state = None
    best_score = float("inf")
    best_epoch = effective_min_epochs
    counter = 0
    history = []
    macro_history: List[float] = []
    started = time.time()
    if progress_prefix:
        _progress(f"{progress_prefix} START | train={len(train_df)} valid={len(valid_df)} max_epochs={max_epochs}")

    for epoch in range(1, max_epochs + 1):
        epoch_t0 = time.time()
        epoch_diag: Dict[str, float] = {}
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            weight_map=weight_map,
            loss_type=str(params.get("loss_type", "mse")),
            huber_delta=float(params.get("huber_delta", 1.0)),
            use_uncertainty_weighting=use_uncertainty_weighting,
            use_pcgrad=use_pcgrad,
            epoch_number=epoch,
            uncertainty_warmup_epochs=uncertainty_warmup_epochs,
            auxiliary_loss_weight=(
                float(params.get("auxiliary_loss_weight", 0.20))
                if use_auxiliary_loss else 0.0
            ),
            rank_loss_weight=float(params.get("rank_loss_weight", 0.0)),
            diagnostics_out=epoch_diag,
        )
        epoch_diag["lc50_aux_scaled_huber"] = np.nan
        epoch_diag["lc50_auxiliary_weight"] = 0.0

        valid_pred = predict_loader(model, valid_loader, device, scaler)
        valid_metrics = regression_metrics(
            valid_pred["true_log10"], valid_pred["pred_log10"]
        )
        macro_score = _macro_task_parent_rmse(valid_pred, valid_df)
        robust_score = _robust_task_parent_score(valid_pred, valid_df)
        macro_history.append(float(robust_score))
        w = max(1, int(smoothing_window))
        smoothed = float(np.mean(macro_history[-w:]))
        scheduler.step(smoothed)

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "valid_RMSE": valid_metrics["RMSE"],
                "valid_R2": valid_metrics["R2"],
                "valid_TaskParentMacroRMSE": macro_score,
                "valid_RobustTaskScore": robust_score,
                "valid_smoothed_RobustTaskScore": smoothed,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "uncertainty_learning_rate": float(optimizer.param_groups[1]["lr"]) if len(optimizer.param_groups) > 1 else np.nan,
                **epoch_diag,
                **_uncertainty_history_fields(model),
            }
        )

        if progress_prefix and (epoch == 1 or epoch % max(1, int(progress_every)) == 0 or epoch == max_epochs):
            weights_now = model.task_uncertainty_table() if hasattr(model, "task_uncertainty_table") else {}
            rel = [v.get("relative_weight", 1.0) for v in weights_now.values()]
            rel_min = min(rel) if rel else 1.0
            rel_max = max(rel) if rel else 1.0
            _progress(
                f"{progress_prefix} epoch {epoch}/{max_epochs} | loss={train_loss:.4f} | "
                f"val_RMSE={valid_metrics['RMSE']:.4f} | val_R2={valid_metrics['R2']:.4f} | "
                f"macroRMSE={macro_score:.4f} robust={robust_score:.4f} | PCGrad conflict={epoch_diag.get('pcgrad_conflict_rate', 0.0):.1%} "
                f"proj={int(epoch_diag.get('pcgrad_projection_events', 0))} | "
                f"uncertainty_active={bool(epoch_diag.get('uncertainty_weighting_active', 0))} "
                f"weight_range={rel_min:.2f}-{rel_max:.2f} | "
                f"epoch_time={(time.time()-epoch_t0)/60:.1f} min"
            )

        
        if epoch < effective_min_epochs:
            continue

        if smoothed < best_score - 1e-5:
            best_score = smoothed
            best_epoch = epoch
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                break

    
    if best_state is not None:
        model.load_state_dict(best_state)

    pred = predict_loader(model, valid_loader, device, scaler)
    if progress_prefix:
        _progress(f"{progress_prefix} DONE | best_epoch={best_epoch} | elapsed={(time.time()-started)/60:.1f} min")
    return model, prep, scaler, pd.DataFrame(history), int(best_epoch), pred

def _refit_full_development(
    development_df: pd.DataFrame,
    params: Dict[str, float | int],
    epochs: int,
    seed: int,
    device: torch.device,
    num_workers: int,
    use_uncertainty_weighting: bool = True,
    use_pcgrad: bool = False,
    uncertainty_lr_multiplier: float = 5.0,
    uncertainty_warmup_epochs: int = 8,
    progress_every: int = 10,
    progress_prefix: str = "Final refit",
    use_cross_task_evidence: bool = True,
    use_record_level_evidence: bool = True,
    use_task_atom_attention: bool = True,
    use_auxiliary_loss: bool = True,
) -> Tuple[
    nn.Module, ContextPreprocessor, TaskTargetScaler,
    CrossTaskEvidenceStore, ContextAwareRecordEvidenceStore, pd.DataFrame
]:
    set_seed(seed)
    prep = ContextPreprocessor(measurement_encoding_mode=_MEASUREMENT_ENCODING_MODE).fit(development_df)
    scaler = TaskTargetScaler.fit(development_df)
    evidence_store = CrossTaskEvidenceStore.fit(
        development_df, scaler, n_tasks=len(TASK_TO_ID)
    )
    record_evidence_store = ContextAwareRecordEvidenceStore.fit(
        development_df,
        prep,
        scaler,
        per_source_task=int(params.get("record_evidence_per_source", 4)),
        n_tasks=len(TASK_TO_ID),
    )
    weight_map = _task_parent_weight_map(
        development_df, power=float(params.get("chemical_weight_power", 0.125))
    )
    loader = _make_loader(
        development_df,
        prep,
        scaler,
        int(params["batch_size"]),
        True,
        num_workers,
        evidence_store=evidence_store,
        record_evidence_store=record_evidence_store,
        use_cross_task_evidence=use_cross_task_evidence,
        use_record_level_evidence=use_record_level_evidence,
        record_evidence_per_source=int(params.get("record_evidence_per_source", 4)),
        task_sampling_power=float(params.get("task_sampling_power", 0.75)),
        seed=seed,
        include_auxiliary_targets=bool(use_auxiliary_loss),
    )
    model = _build_model(
        prep.output_dims,
        params,
        use_cross_task_evidence=use_cross_task_evidence,
        use_record_level_evidence=use_record_level_evidence,
        use_task_atom_attention=use_task_atom_attention,
    ).to(device)
    optimizer = _make_optimizer(
        model,
        lr=float(params["lr"]),
        weight_decay=float(params["weight_decay"]),
        uncertainty_lr_multiplier=uncertainty_lr_multiplier,
    )
    lc50_loader = None  # v10-fast: frozen LC50 expert only.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, int(epochs)), eta_min=1e-6
    )

    hist = []
    started = time.time()
    _progress(f"{progress_prefix} START | records={len(development_df)} epochs={epochs}")
    for epoch in range(1, max(1, int(epochs)) + 1):
        epoch_t0 = time.time()
        epoch_diag: Dict[str, float] = {}
        loss = train_epoch(
            model,
            loader,
            optimizer,
            device,
            weight_map=weight_map,
            loss_type=str(params.get("loss_type", "mse")),
            huber_delta=float(params.get("huber_delta", 1.5)),
            use_uncertainty_weighting=use_uncertainty_weighting,
            use_pcgrad=use_pcgrad,
            epoch_number=epoch,
            uncertainty_warmup_epochs=uncertainty_warmup_epochs,
            auxiliary_loss_weight=(
                float(params.get("auxiliary_loss_weight", 0.20))
                if use_auxiliary_loss else 0.0
            ),
            rank_loss_weight=float(params.get("rank_loss_weight", 0.0)),
            diagnostics_out=epoch_diag,
        )
        epoch_diag["lc50_aux_scaled_huber"] = np.nan
        epoch_diag["lc50_auxiliary_weight"] = 0.0
        hist.append(
            {
                "epoch": epoch,
                "train_loss": loss,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "uncertainty_learning_rate": float(optimizer.param_groups[1]["lr"]) if len(optimizer.param_groups) > 1 else np.nan,
                **epoch_diag,
                **_uncertainty_history_fields(model),
            }
        )
        scheduler.step()
        if epoch == 1 or epoch % max(1, int(progress_every)) == 0 or epoch == epochs:
            table = model.task_uncertainty_table() if hasattr(model, "task_uncertainty_table") else {}
            rel = [v.get("relative_weight", 1.0) for v in table.values()]
            _progress(
                f"{progress_prefix} epoch {epoch}/{epochs} | loss={loss:.4f} | "
                f"PCGrad conflict={epoch_diag.get('pcgrad_conflict_rate', 0.0):.1%} "
                f"proj={int(epoch_diag.get('pcgrad_projection_events', 0))} | "
                f"uncertainty_active={bool(epoch_diag.get('uncertainty_weighting_active', 0))} "
                f"weight_range={(min(rel) if rel else 1.0):.2f}-{(max(rel) if rel else 1.0):.2f} | "
                f"epoch_time={(time.time()-epoch_t0)/60:.1f} min"
            )
    _progress(f"{progress_prefix} DONE | elapsed={(time.time()-started)/60:.1f} min")
    return model, prep, scaler, evidence_store, record_evidence_store, pd.DataFrame(hist)



def suggest_params(
    trial: optuna.Trial,
    use_auxiliary_loss: bool = True,
) -> Dict[str, float | int | str]:
   
    params: Dict[str, float | int | str] = {
        "hidden_dim": 64,
        "n_layers": 5,
        "num_heads": 1,
        "dropout": float(trial.suggest_categorical("dropout", [0.07, 0.09, 0.11])),
        "lr": float(trial.suggest_categorical("lr", [8e-5, 1.0e-4, 1.3e-4])),
        "weight_decay": float(trial.suggest_categorical("weight_decay", [1e-6, 5e-6, 1e-5])),
        "batch_size": 64,
        "chemical_weight_power": 0.25,
        "task_sampling_power": 1.0,
        "record_evidence_per_source": int(
            trial.suggest_categorical("record_evidence_per_source", [1, 2])
        ),
        "record_attention_temperature": float(
            trial.suggest_categorical("record_attention_temperature", [0.25, 0.40, 0.60])
        ),
        "rank_loss_weight": float(
            trial.suggest_categorical("rank_loss_weight", [0.0, 0.03])
        ),
        "loss_type": "mse",
        "huber_delta": 1.5,
    }
    if use_auxiliary_loss:
        params["auxiliary_loss_weight"] = float(
            trial.suggest_categorical(
                "auxiliary_loss_weight", [0.10, 0.20, 0.30]
            )
        )
    else:
        params["auxiliary_loss_weight"] = 0.0
    params["lc50_auxiliary_weight"] = 0.0
    return params


def optimize_hyperparameters(
    development_df: pd.DataFrame,
    n_trials: int,
    hpo_folds: int,
    max_epochs: int,
    patience: int,
    seed: int,
    device: torch.device,
    output_dir: Path,
    num_workers: int,
    use_uncertainty_weighting: bool = True,
    use_pcgrad: bool = False,
    uncertainty_lr_multiplier: float = 5.0,
    uncertainty_warmup_epochs: int = 8,
    progress_every: int = 10,
    initial_params: Optional[Dict[str, object]] = None,
    use_cross_task_evidence: bool = True,
    use_record_level_evidence: bool = True,
    use_task_atom_attention: bool = True,
    use_auxiliary_loss: bool = True,
) -> Tuple[Dict[str, float | int], pd.DataFrame]:
    """Optuna HPO using one fixed set of leakage-safe folds for every trial."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    n_groups = development_df[GROUP_COL].nunique()
    requested_splits = min(max(2, int(hpo_folds)), int(n_groups))
    fixed_splits = task_balanced_group_kfold_split(
        development_df,
        n_splits=requested_splits,
        random_state=seed + 777,
    )
    n_splits = len(fixed_splits)

  
    fold_rows = []
    for fold, (_tr, va) in enumerate(fixed_splits, start=1):
        for pos in va:
            row = development_df.iloc[int(pos)]
            fold_rows.append({
                "fold": fold,
                "frame_index": int(development_df.index[int(pos)]),
                "task": str(row[TASK_COL]),
                "parent": str(row[GROUP_COL]),
            })
    pd.DataFrame(fold_rows).to_csv(
        output_dir / "hpo_fixed_fold_assignments.csv", index=False
    )

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial, use_auxiliary_loss=use_auxiliary_loss)
        _progress(f"HPO trial {trial.number + 1}/{n_trials} START | params={params}")
        scores = []
        for fold, (tr, va) in enumerate(fixed_splits, start=1):
            fold_train = development_df.iloc[tr].copy()
            fold_valid = development_df.iloc[va].copy()
            _model, _prep, _scaler, _hist, _best_epoch, pred = _fit_fold(
                fold_train,
                fold_valid,
                params=params,
                max_epochs=max_epochs,
                patience=patience,
                seed=seed + 1000 * trial.number + fold,
                device=device,
                num_workers=num_workers,
                use_uncertainty_weighting=use_uncertainty_weighting,
                use_pcgrad=use_pcgrad,
                uncertainty_lr_multiplier=uncertainty_lr_multiplier,
                uncertainty_warmup_epochs=uncertainty_warmup_epochs,
                progress_every=progress_every,
                progress_prefix=f"HPO trial {trial.number + 1}/{n_trials} fold {fold}/{n_splits}",
                use_cross_task_evidence=use_cross_task_evidence,
                use_record_level_evidence=use_record_level_evidence,
                use_task_atom_attention=use_task_atom_attention,
                use_auxiliary_loss=use_auxiliary_loss,
            )
            score = _robust_task_parent_score(pred, fold_valid)
            scores.append(score)
            _progress(
                f"HPO trial {trial.number + 1}/{n_trials} fold {fold}/{n_splits} "
                f"score={score:.4f} running_mean={np.mean(scores):.4f}"
            )
            trial.report(float(np.mean(scores)), step=fold)
            if trial.should_prune():
                raise optuna.TrialPruned()
        final_score = float(np.mean(scores))
        _progress(
            f"HPO trial {trial.number + 1}/{n_trials} DONE | mean_robustTaskScore={final_score:.4f}"
        )
        return final_score

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=max(5, n_trials // 5), n_warmup_steps=1
        ),
    )

   
    if initial_params:
        allowed_keys = {
            "dropout", "lr", "weight_decay",
            "record_evidence_per_source", "record_attention_temperature",
            "rank_loss_weight", "auxiliary_loss_weight",
        }
        allowed = {k: v for k, v in initial_params.items() if k in allowed_keys}
        if not use_auxiliary_loss:
            allowed.pop("auxiliary_loss_weight", None)
        legal = (
            int(initial_params.get("hidden_dim", 64)) == 64
            and int(initial_params.get("batch_size", 64)) == 64
            and str(initial_params.get("loss_type", "mse")) == "mse"
            and float(allowed.get("dropout", 0.09)) in (0.07, 0.09, 0.11)
            and float(allowed.get("lr", 1e-4)) in (8e-5, 1e-4, 1.3e-4)
            and float(allowed.get("weight_decay", 5e-6)) in (1e-6, 5e-6, 1e-5)
            and int(allowed.get("record_evidence_per_source", 2)) in (1, 2)
            and float(allowed.get("record_attention_temperature", 0.40)) in (0.25, 0.40, 0.60)
            and float(allowed.get("rank_loss_weight", 0.03)) in (0.0, 0.03)
        )
        if legal and allowed:
            study.enqueue_trial(allowed)
            _progress(f"Warm-start HPO: enqueued compatible v10-fast parameters {allowed}")

    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    trials_df = study.trials_dataframe()
    trials_df.to_csv(output_dir / "optuna_trials.csv", index=False)
    best_params = dict(study.best_params)
    best_params.setdefault("hidden_dim", 64)
    best_params.setdefault("n_layers", 5)
    best_params.setdefault("chemical_weight_power", 0.25)
    best_params.setdefault("num_heads", 1)
    best_params.setdefault("batch_size", 64)
    best_params.setdefault("loss_type", "mse")
    best_params.setdefault("huber_delta", 1.5)
    best_params.setdefault("task_sampling_power", 1.0)
    best_params.setdefault("record_evidence_per_source", 2)
    best_params.setdefault("record_attention_temperature", 0.40)
    best_params.setdefault("rank_loss_weight", 0.03)
    best_params.setdefault("auxiliary_loss_weight", 0.0 if not use_auxiliary_loss else 0.20)
    best_params["lc50_auxiliary_weight"] = 0.0
    with open(output_dir / "best_hyperparameters.json", "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2)
    return best_params, trials_df


def repeated_group_cv(
    development_df: pd.DataFrame,
    params: Dict[str, float | int | str],
    cv_folds: int,
    cv_repeats: int,
    max_epochs: int,
    patience: int,
    seed: int,
    device: torch.device,
    num_workers: int,
    min_epochs: int = 20,
    smoothing_window: int = 5,
    use_uncertainty_weighting: bool = True,
    use_pcgrad: bool = False,
    uncertainty_lr_multiplier: float = 5.0,
    uncertainty_warmup_epochs: int = 8,
    progress_every: int = 10,
    use_cross_task_evidence: bool = True,
    use_record_level_evidence: bool = True,
    use_task_atom_attention: bool = True,
    use_auxiliary_loss: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
   
    folds = min(int(cv_folds), int(development_df[GROUP_COL].nunique()))
    overall_rows = []
    task_rows = []
    oof_rows = []
    total_fits = int(cv_repeats) * folds
    fit_counter = 0
    _progress(f"TASK-TRANSFER CV START | {folds}-fold x {cv_repeats} repeats = {total_fits} fits")

    for repeat in range(int(cv_repeats)):
        splits = task_balanced_group_kfold_split(
            development_df,
            n_splits=folds,
            random_state=seed + repeat,
        )
        for fold, (tr, va) in enumerate(splits, start=1):
            fit_counter += 1
            fold_train = development_df.iloc[tr].copy()
            fold_valid = development_df.iloc[va].copy()
            _model, _prep, _scaler, _hist, best_epoch, pred = _fit_fold(
                fold_train,
                fold_valid,
                params=params,
                max_epochs=max_epochs,
                patience=patience,
                seed=seed + repeat * 100 + fold,
                device=device,
                num_workers=num_workers,
                min_epochs=min_epochs,
                smoothing_window=smoothing_window,
                use_uncertainty_weighting=use_uncertainty_weighting,
                use_pcgrad=use_pcgrad,
                uncertainty_lr_multiplier=uncertainty_lr_multiplier,
                uncertainty_warmup_epochs=uncertainty_warmup_epochs,
                progress_every=progress_every,
                progress_prefix=f"Task-transfer CV {fit_counter}/{total_fits} (repeat {repeat+1}/{cv_repeats}, fold {fold}/{folds})",
                use_cross_task_evidence=use_cross_task_evidence,
                use_record_level_evidence=use_record_level_evidence,
                use_task_atom_attention=use_task_atom_attention,
                use_auxiliary_loss=use_auxiliary_loss,
            )
            overall_metrics = regression_metrics(
                pred["true_log10"], pred["pred_log10"]
            )
            overall_rows.append(
                {
                    "repeat": repeat + 1,
                    "fold": fold,
                    "n_train": len(fold_train),
                    "n_valid": len(fold_valid),
                    "n_train_parent_structures": fold_train[GROUP_COL].nunique(),
                    "n_valid_parent_structures": fold_valid[GROUP_COL].nunique(),
                    "n_valid_tasks": pred["task"].nunique(),
                    "best_epoch": best_epoch,
                    "TaskParentMacroRMSE": _macro_task_parent_rmse(
                        pred, fold_valid
                    ),
                    **overall_metrics,
                }
            )

            pred_fold = pred.copy()
            pred_fold["repeat"] = repeat + 1
            pred_fold["fold"] = fold
           
            meta = (
                fold_valid[[GROUP_COL, TASK_COL]]
                .reset_index()
                .rename(columns={"index": "row_index"})
            )
            pred_fold = pred_fold.merge(meta, on="row_index", how="left")
            oof_rows.extend(pred_fold.to_dict("records"))

            for task, sub in pred.groupby("task"):
                tm = regression_metrics(sub["true_log10"], sub["pred_log10"])
                task_rows.append(
                    {
                        "repeat": repeat + 1,
                        "fold": fold,
                        "task": task,
                        "n_valid": len(sub),
                        "n_valid_parent_structures": fold_valid.loc[
                            fold_valid[TASK_COL] == task, GROUP_COL
                        ].nunique(),
                        **tm,
                    }
                )

    return (
        pd.DataFrame(overall_rows),
        pd.DataFrame(task_rows),
        pd.DataFrame(oof_rows),
    )


def repeated_record_level_cv(
    development_df: pd.DataFrame,
    params: Dict[str, float | int | str],
    cv_folds: int,
    cv_repeats: int,
    max_epochs: int,
    patience: int,
    seed: int,
    device: torch.device,
    num_workers: int,
    min_epochs: int = 20,
    smoothing_window: int = 5,
    use_uncertainty_weighting: bool = True,
    use_pcgrad: bool = False,
    uncertainty_lr_multiplier: float = 5.0,
    uncertainty_warmup_epochs: int = 8,
    progress_every: int = 10,
    use_cross_task_evidence: bool = True,
    use_record_level_evidence: bool = True,
    use_task_atom_attention: bool = True,
    use_auxiliary_loss: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
 
    work = development_df.reset_index(drop=True).copy()
    n_splits = max(2, min(int(cv_folds), int(work[TASK_COL].value_counts().min())))
    overall_rows: List[dict] = []
    task_rows: List[dict] = []
    oof_rows: List[dict] = []

    labels = work[TASK_COL].astype(str).to_numpy()
    total_fits = int(cv_repeats) * n_splits
    fit_counter = 0
    _progress(f"RECORD-LEVEL CV START | {n_splits}-fold x {cv_repeats} repeats = {total_fits} fits")
    for repeat in range(int(cv_repeats)):
        skf = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=seed + 50000 + repeat,
        )
        for fold, (tr, va) in enumerate(skf.split(np.zeros(len(work)), labels), start=1):
            fit_counter += 1
            fold_train = work.iloc[tr].copy()
            fold_valid = work.iloc[va].copy()
            _model, _prep, _scaler, _hist, best_epoch, pred = _fit_fold(
                fold_train,
                fold_valid,
                params=params,
                max_epochs=max_epochs,
                patience=patience,
                seed=seed + 50000 + repeat * 100 + fold,
                device=device,
                num_workers=num_workers,
                min_epochs=min_epochs,
                smoothing_window=smoothing_window,
                use_uncertainty_weighting=use_uncertainty_weighting,
                use_pcgrad=use_pcgrad,
                uncertainty_lr_multiplier=uncertainty_lr_multiplier,
                uncertainty_warmup_epochs=uncertainty_warmup_epochs,
                progress_every=progress_every,
                progress_prefix=f"Record-level CV {fit_counter}/{total_fits} (repeat {repeat+1}/{cv_repeats}, fold {fold}/{n_splits})",
                use_cross_task_evidence=use_cross_task_evidence,
                use_record_level_evidence=use_record_level_evidence,
                use_task_atom_attention=use_task_atom_attention,
                use_auxiliary_loss=use_auxiliary_loss,
            )
            mm = regression_metrics(pred["true_log10"], pred["pred_log10"])
            train_parents = set(fold_train[GROUP_COL].astype(str))
            valid_parents = set(fold_valid[GROUP_COL].astype(str))
            overall_rows.append({
                "repeat": repeat + 1,
                "fold": fold,
                "n_train": len(fold_train),
                "n_valid": len(fold_valid),
                "n_train_parent_structures": len(train_parents),
                "n_valid_parent_structures": len(valid_parents),
                "n_parent_overlap": len(train_parents & valid_parents),
                "best_epoch": best_epoch,
                **mm,
            })

            pred_fold = pred.copy()
            pred_fold["repeat"] = repeat + 1
            pred_fold["fold"] = fold
            meta = (
                fold_valid[[GROUP_COL, TASK_COL]]
                .reset_index()
                .rename(columns={"index": "row_index"})
            )
            pred_fold = pred_fold.merge(meta, on="row_index", how="left")
            oof_rows.extend(pred_fold.to_dict("records"))

            for task, sub in pred.groupby("task"):
                tm = regression_metrics(sub["true_log10"], sub["pred_log10"])
                task_valid = fold_valid[fold_valid[TASK_COL].astype(str) == str(task)]
                task_train = fold_train[fold_train[TASK_COL].astype(str) == str(task)]
                pva = set(task_valid[GROUP_COL].astype(str))
                ptr = set(task_train[GROUP_COL].astype(str))
                task_rows.append({
                    "repeat": repeat + 1,
                    "fold": fold,
                    "task": task,
                    "n_valid": len(sub),
                    "n_valid_parent_structures": len(pva),
                    "n_same_task_parent_overlap": len(pva & ptr),
                    **tm,
                })

    return pd.DataFrame(overall_rows), pd.DataFrame(task_rows), pd.DataFrame(oof_rows)


def summarize_pooled_oof_by_task(
    oof_df: pd.DataFrame,
    output_path: Path,
    prefix: str,
) -> pd.DataFrame:
    """Summarize pooled OOF performance for each repeat and task."""
    rows = []
    if not oof_df.empty:
        for (repeat_id, task), sub in oof_df.groupby(["repeat", "task"]):
            mm = regression_metrics(sub["true_log10"], sub["pred_log10"])
            rows.append({
                "repeat": int(repeat_id),
                "task": task,
                "n_records": len(sub),
                "n_parent_structures": sub[GROUP_COL].nunique() if GROUP_COL in sub.columns else np.nan,
                **mm,
            })
    repeat_df = pd.DataFrame(rows)
    repeat_df.to_csv(output_path / f"{prefix}_repeat_metrics_by_task.csv", index=False)

    summary_rows = []
    if not repeat_df.empty:
        for task, sub in repeat_df.groupby("task"):
            for metric in ["R2", "RMSE", "MAE", "Pearson_r", "Spearman_rho"]:
                vals = pd.to_numeric(sub[metric], errors="coerce").dropna()
                if len(vals):
                    summary_rows.append({
                        "task": task,
                        "metric": metric,
                        "mean": float(vals.mean()),
                        "sd": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                        "median": float(vals.median()),
                        "n_repeats": int(len(vals)),
                    })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_path / f"{prefix}_summary_by_task.csv", index=False)
    return summary_df



def fit_oof_task_calibration(
    oof_df: pd.DataFrame,
    strength: float = 50.0,
) -> pd.DataFrame:
  
    columns = [
        "task", "n_rows", "raw_slope", "slope", "intercept",
        "prediction_mean", "target_mean", "strength"
    ]
    if oof_df.empty:
        return pd.DataFrame(columns=columns)

    keys = ["row_index", "task", "true_log10"]
    averaged = (
        oof_df.groupby(keys, as_index=False)["pred_log10"].mean()
        .rename(columns={"pred_log10": "oof_pred_mean"})
    )
    rows = []
    for task, sub in averaged.groupby("task"):
        x = sub["oof_pred_mean"].to_numpy(dtype=float)
        y = sub["true_log10"].to_numpy(dtype=float)
        n = len(sub)
        x_mean = float(np.mean(x)) if n else 0.0
        y_mean = float(np.mean(y)) if n else 0.0
        var_x = float(np.mean((x - x_mean) ** 2)) if n else 0.0
        cov_xy = float(np.mean((x - x_mean) * (y - y_mean))) if n else 0.0
        raw_slope = cov_xy / var_x if var_x > 1e-10 else 1.0
        reliability = float(n / (n + max(float(strength), 1e-8)))
        slope = 1.0 + reliability * (raw_slope - 1.0)
        slope = float(np.clip(slope, 0.50, 2.50))
        intercept = float(y_mean - slope * x_mean)
        rows.append({
            "task": str(task),
            "n_rows": int(n),
            "raw_slope": float(raw_slope),
            "slope": slope,
            "intercept": intercept,
            "prediction_mean": x_mean,
            "target_mean": y_mean,
            "strength": float(strength),
        })
    return pd.DataFrame(rows, columns=columns)


def apply_task_calibration(
    pred_df: pd.DataFrame,
    calibration_df: pd.DataFrame,
) -> pd.DataFrame:
    out = pred_df.copy()
    out["pred_log10_raw"] = out["pred_log10"].astype(float)
    if "pred_log10_sd" in out.columns:
        out["pred_log10_sd_raw"] = out["pred_log10_sd"].astype(float)

    if calibration_df.empty:
        out["calibration_slope"] = 1.0
        out["calibration_intercept"] = 0.0
        return out

    cal = calibration_df.set_index("task")
    slopes = []
    intercepts = []
    calibrated = []
    calibrated_sd = []
    has_sd = "pred_log10_sd" in out.columns
    for _, row in out.iterrows():
        task = str(row["task"])
        if task in cal.index:
            slope = float(cal.at[task, "slope"])
            intercept = float(cal.at[task, "intercept"])
        else:
            slope, intercept = 1.0, 0.0
        raw = float(row["pred_log10_raw"])
        slopes.append(slope)
        intercepts.append(intercept)
        calibrated.append(intercept + slope * raw)
        if has_sd:
            calibrated_sd.append(abs(slope) * float(row["pred_log10_sd_raw"]))

    out["calibration_slope"] = slopes
    out["calibration_intercept"] = intercepts
    out["pred_log10"] = np.asarray(calibrated, dtype=float)
    if has_sd:
        out["pred_log10_sd"] = np.asarray(calibrated_sd, dtype=float)
    out["pred_mg_L"] = np.power(10.0, out["pred_log10"].to_numpy(dtype=float))
    return out


def choose_final_epoch(
    development_df: pd.DataFrame,
    params: Dict[str, float | int],
    max_epochs: int,
    patience: int,
    seed: int,
    device: torch.device,
    num_workers: int,
    use_uncertainty_weighting: bool = True,
    use_pcgrad: bool = True,
) -> Tuple[int, pd.DataFrame]:
  
    splits = task_balanced_group_kfold_split(
        development_df,
        n_splits=min(8, development_df[GROUP_COL].nunique()),
        random_state=seed,
    )
    tr, va = splits[0]
    _model, _prep, _scaler, hist, best_epoch, _pred = _fit_fold(
        development_df.iloc[tr].copy(),
        development_df.iloc[va].copy(),
        params=params,
        max_epochs=max_epochs,
        patience=patience,
        seed=seed,
        device=device,
        num_workers=num_workers,
        use_uncertainty_weighting=use_uncertainty_weighting,
        use_pcgrad=use_pcgrad,
    )
    return int(best_epoch), hist



def run_multitask_training(
    combined_df: pd.DataFrame,
    output_dir: str | Path,
    hpo_trials: int = 30,
    hpo_folds: int = 3,
    cv_folds: int = 10,
    cv_repeats: int = 2,
    max_epochs: int = 200,
    patience: int = 20,
    seed: int = DEFAULT_RANDOM_STATE,
    device_name: str = "auto",
    num_workers: int = 0,
    final_ensemble_seeds: int = 5,
    run_record_level_cv: bool = False,
    record_cv_folds: int = 10,
    record_cv_repeats: int = 1,
    use_uncertainty_weighting: bool = True,
    use_pcgrad: bool = False,
    uncertainty_lr_multiplier: float = 5.0,
    uncertainty_warmup_epochs: int = 8,
    progress_every: int = 10,
    initial_hpo_params: Optional[Dict[str, object]] = None,
    fixed_params: Optional[Dict[str, object]] = None,
    use_cross_task_evidence: bool = True,
    use_record_level_evidence: bool = True,
    use_task_atom_attention: bool = True,
    use_auxiliary_loss: bool = True,
    use_oof_calibration: bool = True,
    calibration_strength: float = 50.0,
    lc50_data_path: Optional[str] = None,
    lc50_pretrained_checkpoint: Optional[str] = None,
    lc50_frozen_expert_checkpoint: Optional[str] = None,
    use_lc50_auxiliary: bool = True,
    lc50_pretrain_epochs: int = 40,
    lc50_pretrain_patience: int = 6,
    lc50_batch_size: int = 128,
    lc50_joint_steps: int = 4,
    use_fingerprint_expert: bool = True,
    use_evidence_attention: bool = True,
    use_record_quality_gate: bool = True,
    use_task_residual: bool = True,
    measurement_encoding_mode: str = "hybrid",
) -> Dict[str, object]:
  
    global _LC50_FRAME, _LC50_PREP, _LC50_MODULE_STATES
    global _LC50_BATCH_SIZE, _LC50_JOINT_STEPS, _LC50_NUM_WORKERS
    global _MEASUREMENT_ENCODING_MODE, _USE_FINGERPRINT_EXPERT
    global _USE_EVIDENCE_ATTENTION, _USE_RECORD_QUALITY_GATE, _USE_TASK_RESIDUAL

    _MEASUREMENT_ENCODING_MODE = str(measurement_encoding_mode).strip().lower()
    if _MEASUREMENT_ENCODING_MODE not in {"hybrid", "exact_only"}:
        raise ValueError("measurement_encoding_mode must be 'hybrid' or 'exact_only'")
    _USE_FINGERPRINT_EXPERT = bool(use_fingerprint_expert)
    _USE_EVIDENCE_ATTENTION = bool(use_evidence_attention)
    _USE_RECORD_QUALITY_GATE = bool(use_record_quality_gate)
    _USE_TASK_RESIDUAL = bool(use_task_residual)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(device_name)

 
    _LC50_FRAME = None
    _LC50_PREP = None
    _LC50_MODULE_STATES = None
    _LC50_BATCH_SIZE = int(lc50_batch_size)
    _LC50_JOINT_STEPS = int(lc50_joint_steps)
    _LC50_NUM_WORKERS = int(num_workers)
    
    if bool(use_lc50_auxiliary) and lc50_frozen_expert_checkpoint:
        frozen_path = Path(lc50_frozen_expert_checkpoint)
        if not frozen_path.is_file():
            raise FileNotFoundError(f"Frozen LC50 weights not found: {frozen_path}")
        frozen = torch.load(frozen_path, map_location="cpu", weights_only=True)
        if frozen.get("format") != "ctml_frozen_lc50_expert_v1":
            raise ValueError("Unrecognized frozen LC50 expert checkpoint")
        _LC50_MODULE_STATES = frozen["module_states"]
     
        _LC50_PREP = SimpleNamespace(species_vocab_size=int(frozen["species_vocab_size"]))
        _progress(f"Frozen LC50 structural expert loaded: {frozen_path}")
    elif bool(use_lc50_auxiliary) and lc50_data_path:
        ckpt_path = Path(lc50_pretrained_checkpoint) if lc50_pretrained_checkpoint else None
        if ckpt_path is not None and ckpt_path.exists():
       
            _LC50_MODULE_STATES, _LC50_PREP, _lc50_best_epoch = load_lc50_pretrained_checkpoint(ckpt_path)
            _progress(f"LC50 PRETRAIN REUSED | checkpoint={ckpt_path} selected_epoch={_lc50_best_epoch}")
        else:
            _progress(f"LC50 AUXILIARY LOAD | {lc50_data_path}")
            lc50_cur = load_and_curate_lc50_fish(lc50_data_path)
            _LC50_FRAME = lc50_cur.frame.copy()
            lc50_cur.audit.to_csv(output_dir / "lc50_curation_audit.csv", index=False)
            _progress(
                f"LC50 PRETRAIN START | records={len(_LC50_FRAME)} "
                f"species={_LC50_FRAME['Species'].nunique()} parents={_LC50_FRAME['ParentSMILES'].nunique()}"
            )
            _LC50_MODULE_STATES, _LC50_PREP, _lc50_hist, _lc50_best_epoch = pretrain_lc50_backbone(
                _LC50_FRAME, output_dir=output_dir, device=device, hidden_dim=64,
                n_layers=5, dropout=0.10, batch_size=int(lc50_batch_size),
                max_epochs=int(lc50_pretrain_epochs), patience=int(lc50_pretrain_patience),
                seed=int(seed), num_workers=int(num_workers),
            )
            _progress(f"LC50 PRETRAIN COMPLETE | selected_epoch={_lc50_best_epoch}")

    development_df = combined_df[combined_df["split"] != "external"].copy()
    external_df = combined_df[combined_df["split"] == "external"].copy()
    if development_df.empty or external_df.empty:
        raise ValueError("Development and external partitions must both be non-empty.")

    if _LC50_FRAME is not None:
        lc50_parents = set(_LC50_FRAME["ParentSMILES"].astype(str))
        dev_parents = set(development_df[GROUP_COL].astype(str))
        ext_parents = set(external_df[GROUP_COL].astype(str))
        overlap_rows = [{
            "lc50_records": int(len(_LC50_FRAME)),
            "lc50_parent_structures": int(len(lc50_parents)),
            "sublethal_development_parent_structures": int(len(dev_parents)),
            "sublethal_external_parent_structures": int(len(ext_parents)),
            "lc50_overlap_with_development_parents": int(len(lc50_parents & dev_parents)),
            "lc50_overlap_with_external_parents": int(len(lc50_parents & ext_parents)),
            "fraction_external_parents_with_lc50": float(len(lc50_parents & ext_parents) / max(len(ext_parents), 1)),
        }]
        pd.DataFrame(overlap_rows).to_csv(output_dir / "lc50_sublethal_overlap_audit.csv", index=False)
        task_lc50_rows = []
        for task, sub in external_df.groupby(TASK_COL):
            parents_t = set(sub[GROUP_COL].astype(str))
            task_lc50_rows.append({
                "task": str(task),
                "external_records": int(len(sub)),
                "external_parent_structures": int(len(parents_t)),
                "external_parents_with_fish_lc50": int(len(parents_t & lc50_parents)),
                "fraction_external_parents_with_fish_lc50": float(len(parents_t & lc50_parents) / max(len(parents_t), 1)),
            })
        pd.DataFrame(task_lc50_rows).to_csv(output_dir / "lc50_external_support_by_task.csv", index=False)

   
    audit_rows = []
    for task in sorted(combined_df[TASK_COL].dropna().astype(str).unique()):
        task_dev = development_df[development_df[TASK_COL].astype(str) == task]
        task_ext = external_df[external_df[TASK_COL].astype(str) == task]
        ext_parents = set(task_ext[GROUP_COL].astype(str))
        same_task_dev = set(task_dev[GROUP_COL].astype(str))
        overlap = ext_parents & same_task_dev
        if overlap:
            raise ValueError(
                f"Task-specific external leakage for {task}: {len(overlap)} parent structures overlap development."
            )
        other_task_dev = set(
            development_df.loc[
                development_df[TASK_COL].astype(str) != task, GROUP_COL
            ].astype(str)
        )
        seen_elsewhere = ext_parents & other_task_dev
        audit_rows.append({
            "task": task,
            "development_records": len(task_dev),
            "external_records": len(task_ext),
            "development_parent_structures": task_dev[GROUP_COL].nunique(),
            "external_parent_structures": len(ext_parents),
            "same_task_external_development_overlap": 0,
            "external_parents_seen_in_other_task_development": len(seen_elsewhere),
            "fraction_external_parents_seen_in_other_task_development": (
                len(seen_elsewhere) / max(len(ext_parents), 1)
            ),
            "external_parents_globally_unseen": len(ext_parents - other_task_dev),
        })
    split_audit = pd.DataFrame(audit_rows)
    split_audit.to_csv(
        output_dir / "multitask_task_transfer_holdout_audit.csv", index=False
    )

    with open(output_dir / "validation_design.json", "w", encoding="utf-8") as f:
        json.dump({
            "version": "v10-fast",
            "design": "v10-fast task-specific chemical holdout with hybrid measurement encoding, ECFP + sublethal RGCN + frozen LC50 structural experts, and selective cross-task evidence",
            "same_task_external_parent_overlap_allowed": False,
            "other_task_external_parent_overlap_allowed": True,
            "interpretation": (
                "External metrics evaluate a chemical unseen in the target task while allowing "
                "training-only observed outcomes for the same chemical in other tasks to serve as evidence."
            ),
            "primary_cv": (
                "task-specific parent-structure grouped CV; each fold rebuilds cross-task evidence "
                "from fold-training rows only"
            ),
            "cross_task_evidence_enabled": bool(use_cross_task_evidence),
            "cross_task_evidence_features": list(TRANSFER_FEATURE_NAMES),
            "context_aware_record_evidence_enabled": bool(use_cross_task_evidence and use_record_level_evidence),
            "context_aware_record_evidence_features": list(RECORD_TRANSFER_FEATURE_NAMES),
            "target_task_evidence_always_masked": True,
            "task_conditioned_modality_gating": True,
            "effect_measurement_encoding": (
                "exact category only" if _MEASUREMENT_ENCODING_MODE == "exact_only"
                else "exact category + char 3-5 gram hashing + word 1-2 gram hashing"
            ),
            "measurement_aware_record_retrieval": True,
            "ecfp4_structural_expert_enabled": bool(_USE_FINGERPRINT_EXPERT),
            "evidence_attention_weighting_enabled": bool(_USE_EVIDENCE_ATTENTION),
            "sample_specific_record_evidence_quality_gate": bool(_USE_RECORD_QUALITY_GATE),
            "fish_lc50_auxiliary_enabled": bool(_LC50_MODULE_STATES is not None),
            "fish_lc50_usage": "chemical-disjoint pretraining of an independent frozen LC50 structural expert; no joint LC50 loss and no direct LC50 value input",
            "fish_lc50_may_include_sublethal_validation_chemicals": True,
            "fish_lc50_interpretation": "declared cross-endpoint auxiliary transfer; no sublethal validation/external labels are used",
            "fish_lc50_duration_unit": "day",
            "fish_lc50_target_unit": "log10(mg/L), already log-transformed",
            "target_conditioned_atom_attention": bool(use_task_atom_attention),
            "masked_auxiliary_molecular_multitask_loss": bool(use_auxiliary_loss),
            "shared_plus_task_residual_decoder": bool(_USE_TASK_RESIDUAL),
            "task_balanced_mixed_batches": True,
            "uncertainty_based_dynamic_task_weighting": bool(use_uncertainty_weighting),
            "pcgrad_shared_backbone": bool(use_pcgrad),
            "development_oof_calibration": bool(use_oof_calibration),
            "calibration_strength": float(calibration_strength),
        }, f, indent=2)

   
    if fixed_params is not None:
        best_params = dict(fixed_params)
        best_params.setdefault("hidden_dim", 64)
        best_params.setdefault("n_layers", 5)
        best_params.setdefault("num_heads", 1)
        best_params.setdefault("batch_size", 64)
        best_params.setdefault("loss_type", "mse")
        best_params.setdefault("huber_delta", 1.5)
        best_params.setdefault("task_sampling_power", 1.0)
        best_params.setdefault("chemical_weight_power", 0.25)
        best_params.setdefault("record_evidence_per_source", 2)
        best_params.setdefault("record_attention_temperature", 0.40)
        best_params.setdefault("rank_loss_weight", 0.03)
        best_params.setdefault("auxiliary_loss_weight", 0.20 if use_auxiliary_loss else 0.0)
        best_params["lc50_auxiliary_weight"] = 0.0
        if not use_auxiliary_loss:
            best_params["auxiliary_loss_weight"] = 0.0
        trials_df = pd.DataFrame()
        with open(output_dir / "best_hyperparameters.json", "w", encoding="utf-8") as f:
            json.dump(best_params, f, indent=2)
        _progress(f"HPO SKIPPED: using fixed parameters {best_params}")
    else:
        _progress(
            f"HPO START | trials={hpo_trials}, fixed_folds={hpo_folds}, "
            f"cross_task_evidence={use_cross_task_evidence}, "
            f"record_evidence={use_record_level_evidence}, aux={use_auxiliary_loss}, "
            f"task_atom={use_task_atom_attention}, PCGrad={use_pcgrad}"
        )
        best_params, trials_df = optimize_hyperparameters(
            development_df,
            n_trials=hpo_trials,
            hpo_folds=hpo_folds,
            max_epochs=max_epochs,
            patience=patience,
            seed=seed,
            device=device,
            output_dir=output_dir,
            num_workers=num_workers,
            use_uncertainty_weighting=use_uncertainty_weighting,
            use_pcgrad=use_pcgrad,
            uncertainty_lr_multiplier=uncertainty_lr_multiplier,
            uncertainty_warmup_epochs=uncertainty_warmup_epochs,
            progress_every=progress_every,
            initial_params=initial_hpo_params,
            use_cross_task_evidence=use_cross_task_evidence,
            use_record_level_evidence=use_record_level_evidence,
            use_task_atom_attention=use_task_atom_attention,
            use_auxiliary_loss=use_auxiliary_loss,
        )
        _progress(f"HPO COMPLETE | best_params={best_params}")


    cv_df, cv_task_df, oof_df = repeated_group_cv(
        development_df,
        params=best_params,
        cv_folds=cv_folds,
        cv_repeats=cv_repeats,
        max_epochs=max_epochs,
        patience=patience,
        seed=seed,
        device=device,
        num_workers=num_workers,
        min_epochs=25,
        smoothing_window=5,
        use_uncertainty_weighting=use_uncertainty_weighting,
        use_pcgrad=use_pcgrad,
        uncertainty_lr_multiplier=uncertainty_lr_multiplier,
        uncertainty_warmup_epochs=uncertainty_warmup_epochs,
        progress_every=progress_every,
        use_cross_task_evidence=use_cross_task_evidence,
        use_record_level_evidence=use_record_level_evidence,
        use_task_atom_attention=use_task_atom_attention,
        use_auxiliary_loss=use_auxiliary_loss,
    )
    cv_df.to_csv(output_dir / "grouped_cross_validation_folds_overall.csv", index=False)
    cv_task_df.to_csv(output_dir / "grouped_cross_validation_folds_by_task.csv", index=False)
    oof_df.to_csv(output_dir / "grouped_cross_validation_oof_predictions.csv", index=False)
    cv_df.to_csv(output_dir / "task_transfer_cross_validation_folds_overall.csv", index=False)
    cv_task_df.to_csv(output_dir / "task_transfer_cross_validation_folds_by_task.csv", index=False)
    oof_df.to_csv(output_dir / "task_transfer_cross_validation_oof_predictions.csv", index=False)
    summarize_metric_table(cv_df).to_csv(
        output_dir / "grouped_cross_validation_summary_overall.csv", index=False
    )

    if not cv_task_df.empty:
        task_summaries = []
        for task, sub in cv_task_df.groupby("task"):
            for metric in ["R2", "RMSE", "MAE"]:
                vals = pd.to_numeric(sub[metric], errors="coerce").dropna()
                if len(vals):
                    task_summaries.append({
                        "task": task,
                        "metric": metric,
                        "mean": float(vals.mean()),
                        "sd": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                        "median": float(vals.median()),
                        "n_folds": int(len(vals)),
                    })
        pd.DataFrame(task_summaries).to_csv(
            output_dir / "grouped_cross_validation_summary_by_task_foldwise.csv",
            index=False,
        )

    pooled_summary = summarize_pooled_oof_by_task(
        oof_df, output_dir, prefix="grouped_cross_validation"
    )
    # Explicit task-transfer aliases.
    repeat_metrics = pd.read_csv(
        output_dir / "grouped_cross_validation_repeat_metrics_by_task.csv"
    )
    repeat_metrics.to_csv(
        output_dir / "task_transfer_cross_validation_repeat_metrics_by_task.csv",
        index=False,
    )
    pooled_summary.to_csv(
        output_dir / "task_transfer_cross_validation_summary_by_task.csv",
        index=False,
    )

   
    calibration_df = (
        fit_oof_task_calibration(oof_df, strength=calibration_strength)
        if use_oof_calibration
        else pd.DataFrame()
    )
    calibration_df.to_csv(output_dir / "oof_task_calibration.csv", index=False)

   
    if run_record_level_cv:
        rec_cv_df, rec_task_df, rec_oof_df = repeated_record_level_cv(
            development_df,
            params=best_params,
            cv_folds=record_cv_folds,
            cv_repeats=record_cv_repeats,
            max_epochs=max_epochs,
            patience=patience,
            seed=seed,
            device=device,
            num_workers=num_workers,
            min_epochs=25,
            smoothing_window=5,
            use_uncertainty_weighting=use_uncertainty_weighting,
            use_pcgrad=use_pcgrad,
            uncertainty_lr_multiplier=uncertainty_lr_multiplier,
            uncertainty_warmup_epochs=uncertainty_warmup_epochs,
            progress_every=progress_every,
            use_cross_task_evidence=use_cross_task_evidence,
            use_record_level_evidence=use_record_level_evidence,
            use_task_atom_attention=use_task_atom_attention,
            use_auxiliary_loss=use_auxiliary_loss,
        )
        rec_cv_df.to_csv(
            output_dir / "record_level_cross_validation_folds_overall.csv", index=False
        )
        rec_task_df.to_csv(
            output_dir / "record_level_cross_validation_folds_by_task.csv", index=False
        )
        rec_oof_df.to_csv(
            output_dir / "record_level_cross_validation_oof_predictions.csv", index=False
        )
        summarize_metric_table(rec_cv_df).to_csv(
            output_dir / "record_level_cross_validation_summary_overall.csv", index=False
        )
        summarize_pooled_oof_by_task(
            rec_oof_df, output_dir, prefix="record_level_cross_validation"
        )


    finite_epochs = pd.to_numeric(cv_df["best_epoch"], errors="coerce").dropna()
    if len(finite_epochs):
        median_epoch = float(finite_epochs.median())
        q75_epoch = float(finite_epochs.quantile(0.75))
        best_epoch = int(np.clip(round(q75_epoch), 30, max_epochs))
    else:
        median_epoch = np.nan
        q75_epoch = np.nan
        best_epoch = min(max(30, max_epochs // 3), max_epochs)
    pd.DataFrame([{
        "selection_method": "CV best-epoch 75th percentile, min 30",
        "final_refit_epoch": best_epoch,
        "cv_best_epoch_median": median_epoch,
        "cv_best_epoch_q25": float(finite_epochs.quantile(0.25)) if len(finite_epochs) else np.nan,
        "cv_best_epoch_q75": q75_epoch,
    }]).to_csv(output_dir / "final_epoch_selection.csv", index=False)

 
    ensemble_n = max(1, int(final_ensemble_seeds))
    models = []
    refit_histories = []
    prep = None
    scaler = None
    evidence_store = None
    record_evidence_store = None
    for ens_i in range(ensemble_n):
        model_i, prep_i, scaler_i, evidence_i, record_evidence_i, hist_i = _refit_full_development(
            development_df,
            params=best_params,
            epochs=best_epoch,
            seed=seed + 1234 + ens_i * 101,
            device=device,
            num_workers=num_workers,
            use_uncertainty_weighting=use_uncertainty_weighting,
            use_pcgrad=use_pcgrad,
            uncertainty_lr_multiplier=uncertainty_lr_multiplier,
            uncertainty_warmup_epochs=uncertainty_warmup_epochs,
            progress_every=progress_every,
            progress_prefix=f"Final ensemble member {ens_i + 1}/{ensemble_n}",
            use_cross_task_evidence=use_cross_task_evidence,
            use_record_level_evidence=use_record_level_evidence,
            use_task_atom_attention=use_task_atom_attention,
            use_auxiliary_loss=use_auxiliary_loss,
        )
        models.append(model_i)
        refit_histories.append(hist_i.assign(ensemble_member=ens_i + 1))
        if prep is None:
            prep = prep_i
            scaler = scaler_i
            evidence_store = evidence_i
            record_evidence_store = record_evidence_i

    final_history_df = pd.concat(refit_histories, ignore_index=True)
    final_history_df.to_csv(output_dir / "final_refit_history.csv", index=False)
    diag_cols = [c for c in [
        "pcgrad_pair_checks", "pcgrad_conflicts", "pcgrad_conflict_rate",
        "pcgrad_projection_events", "pcgrad_mean_cosine", "pcgrad_mean_negative_cosine"
    ] if c in final_history_df.columns]
    if diag_cols:
        summary_rows = []
        for member, sub_hist in final_history_df.groupby("ensemble_member"):
            row = {"ensemble_member": int(member)}
            for c in diag_cols:
                vals = pd.to_numeric(sub_hist[c], errors="coerce").dropna()
                row[c + "_mean"] = float(vals.mean()) if len(vals) else np.nan
                row[c + "_sum"] = float(vals.sum()) if len(vals) else np.nan
            summary_rows.append(row)
        pd.DataFrame(summary_rows).to_csv(
            output_dir / "pcgrad_diagnostics_final_refit.csv", index=False
        )

    assert (
        prep is not None and scaler is not None and evidence_store is not None
        and record_evidence_store is not None
    )
    prep.save(output_dir / "context_preprocessor.joblib")
    joblib.dump(evidence_store, output_dir / "cross_task_evidence_store.joblib")
    
    evidence_store.support_summary(development_df).to_csv(
        output_dir / "cross_task_evidence_support_development.csv", index=False
    )
    evidence_store.support_summary(external_df).to_csv(
        output_dir / "cross_task_evidence_support_external.csv", index=False
    )
    dev_context_for_support = prep.transform(development_df)
    ext_context_for_support = prep.transform(external_df)
    record_evidence_store.support_summary(
        development_df, dev_context_for_support
    ).to_csv(
        output_dir / "record_evidence_support_development.csv", index=False
    )
    record_evidence_store.support_summary(
        external_df, ext_context_for_support
    ).to_csv(
        output_dir / "record_evidence_support_external.csv", index=False
    )

    
    overall_rows = []
    task_rows = []
    overall_raw_rows = []
    task_raw_rows = []

    for split_name, frame in [
        ("development", development_df),
        ("external", external_df),
    ]:
        loader = _make_loader(
            frame,
            prep,
            scaler,
            int(best_params["batch_size"]),
            False,
            num_workers,
            evidence_store=evidence_store,
            record_evidence_store=record_evidence_store,
            use_cross_task_evidence=use_cross_task_evidence,
            use_record_level_evidence=use_record_level_evidence,
            record_evidence_per_source=int(best_params.get("record_evidence_per_source", 4)),
            task_sampling_power=float(best_params.get("task_sampling_power", 1.0)),
            seed=seed,
        )

        member_predictions = []
        for ens_i, model_i in enumerate(models, start=1):
            pi = predict_loader(
                model_i, loader, device, scaler, return_attention=(ens_i == 1)
            ).sort_values("row_index").reset_index(drop=True)
            member_predictions.append(pi)

        pred_raw = member_predictions[0].copy()
        stack = np.stack(
            [x["pred_log10"].to_numpy(dtype=float) for x in member_predictions],
            axis=1,
        )
        pred_raw["pred_log10"] = stack.mean(axis=1)
        pred_raw["pred_log10_sd"] = stack.std(axis=1, ddof=0)
        pred_raw["pred_mg_L"] = np.power(
            10.0, pred_raw["pred_log10"].to_numpy(dtype=float)
        )

        pred = (
            apply_task_calibration(pred_raw, calibration_df)
            if use_oof_calibration
            else apply_task_calibration(pred_raw, pd.DataFrame())
        )

        meta_cols = [
            "RecordID", "Chemicals", GROUP_COL, "Species", "Family",
            "Environment", "Effect Measurement", "Response Site", "Endpoint",
            "Exposure time,day", "Citation", TASK_COL,
        ]
        available = [c for c in meta_cols if c in frame.columns]
        meta = frame.loc[:, available].reset_index(drop=False).rename(
            columns={"index": "row_index"}
        )
        pred = pred.merge(meta, on="row_index", how="left")
        pred.to_csv(output_dir / f"predictions_{split_name}.csv", index=False)

        overall_raw_rows.append(metric_row(split_name, pred_raw, frame))
        overall_rows.append(metric_row(split_name, pred, frame))
        for task, sub in pred.groupby("task"):
            original_sub = frame[frame[TASK_COL].astype(str) == str(task)]
            task_rows.append({
                "split": split_name,
                "task": task,
                "n_records": len(sub),
                "n_parent_structures": original_sub[GROUP_COL].nunique(),
                **regression_metrics(sub["true_log10"], sub["pred_log10"]),
            })
        for task, sub in pred_raw.groupby("task"):
            original_sub = frame[frame[TASK_COL].astype(str) == str(task)]
            task_raw_rows.append({
                "split": split_name,
                "task": task,
                "n_records": len(sub),
                "n_parent_structures": original_sub[GROUP_COL].nunique(),
                **regression_metrics(sub["true_log10"], sub["pred_log10"]),
            })

    overall_df = pd.DataFrame(overall_rows)
    task_df = pd.DataFrame(task_rows)
    overall_raw_df = pd.DataFrame(overall_raw_rows)
    task_raw_df = pd.DataFrame(task_raw_rows)
    overall_df.to_csv(output_dir / "final_split_metrics_overall.csv", index=False)
    task_df.to_csv(output_dir / "final_split_metrics_by_task.csv", index=False)
    overall_raw_df.to_csv(output_dir / "final_split_metrics_overall_raw.csv", index=False)
    task_raw_df.to_csv(output_dir / "final_split_metrics_by_task_raw.csv", index=False)

    with open(output_dir / "task_target_scalers.json", "w", encoding="utf-8") as f:
        json.dump(scaler.to_dict(), f, indent=2)

    
    uncertainty_rows = []
    for ens_i, model_i in enumerate(models, start=1):
        if hasattr(model_i, "task_uncertainty_table"):
            table = model_i.task_uncertainty_table()
            for task_id, stats in table.items():
                uncertainty_rows.append({
                    "ensemble_member": ens_i,
                    "task_id": int(task_id),
                    "task": ID_TO_TASK[int(task_id)],
                    **stats,
                })
    if uncertainty_rows:
        uncertainty_df = pd.DataFrame(uncertainty_rows)
        uncertainty_df.to_csv(
            output_dir / "learned_task_uncertainty_weights_by_ensemble.csv",
            index=False,
        )
        mean_cols = [
            c for c in [
                "log_variance", "raw_precision_weight", "relative_weight", "residual_scale", "task_atom_scale", "record_evidence_scale"
            ] if c in uncertainty_df.columns
        ]
        (
            uncertainty_df.groupby(["task_id", "task"], as_index=False)[mean_cols]
            .mean()
            .to_csv(
                output_dir / "learned_task_uncertainty_weights_mean.csv", index=False
            )
        )

   
    compat_rows = []
    for ens_i, model_i in enumerate(models, start=1):
        if hasattr(model_i, "record_source_target_bias"):
            mat = model_i.record_source_target_bias.detach().cpu().numpy()
            for target_id in range(mat.shape[0]):
                for source_id in range(mat.shape[1]):
                    compat_rows.append({
                        "ensemble_member": ens_i,
                        "target_task": ID_TO_TASK[target_id],
                        "source_task": ID_TO_TASK[source_id],
                        "compatibility_logit": float(mat[target_id, source_id]),
                    })
    if compat_rows:
        pd.DataFrame(compat_rows).to_csv(
            output_dir / "record_source_target_compatibility_by_ensemble.csv", index=False
        )

    torch.save(
        {
            "architecture": "v10_fast_multi_expert_lc50_ecfp_selective_evidence",
            "state_dicts": [m.state_dict() for m in models],
            "ensemble_size": ensemble_n,
            "best_params": best_params,
            "best_epoch": best_epoch,
            "context_dims": prep.output_dims,
            "tasks": TASK_TO_ID,
            "target": TARGET_LOG_COL,
            "task_target_scalers": scaler.to_dict(),
            "transfer_feature_names": list(TRANSFER_FEATURE_NAMES),
            "record_transfer_feature_names": list(RECORD_TRANSFER_FEATURE_NAMES),
            "use_record_level_evidence": bool(use_record_level_evidence),
            "use_task_atom_attention": bool(use_task_atom_attention),
            "use_auxiliary_loss": bool(use_auxiliary_loss),
            "use_cross_task_evidence": bool(use_cross_task_evidence),
            "use_oof_calibration": bool(use_oof_calibration),
            "oof_calibration": calibration_df.to_dict("records"),
            "use_uncertainty_weighting": bool(use_uncertainty_weighting),
            "use_pcgrad": bool(use_pcgrad),
            "use_fingerprint_expert": bool(_USE_FINGERPRINT_EXPERT),
            "use_evidence_attention": bool(_USE_EVIDENCE_ATTENTION),
            "use_record_quality_gate": bool(_USE_RECORD_QUALITY_GATE),
            "use_task_residual": bool(_USE_TASK_RESIDUAL),
            "measurement_encoding_mode": _MEASUREMENT_ENCODING_MODE,
        },
        output_dir / "multitask_model.pt",
    )

    return {
        "best_params": best_params,
        "best_epoch": best_epoch,
        "overall_metrics": overall_df,
        "task_metrics": task_df,
        "overall_metrics_raw": overall_raw_df,
        "task_metrics_raw": task_raw_df,
        "calibration": calibration_df,
    }

