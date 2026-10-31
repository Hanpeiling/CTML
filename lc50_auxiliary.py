from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
import copy
import math
import random

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from graph_utils import MoleculeGraph, smiles_to_graph
from preprocessing import standardize_smiles
from models import LC50PretrainRegressor

LC50_SPECIES_COL = "Test organisms (species)"
LC50_DURATION_COL = "Duration.MeanValue"  
LC50_TARGET_COL = "Value.MeanValue"       
LC50_SMILES_COL = "SMILES"
LC50_CAS_COL = "CAS Number"


@dataclass
class LC50CurationResult:
    frame: pd.DataFrame
    audit: pd.DataFrame


@dataclass
class LC50Preprocessor:
    species_to_id: Dict[str, int]
    duration_log_mean: float
    duration_log_std: float
    target_mean: float
    target_std: float

    @classmethod
    def fit(cls, frame: pd.DataFrame, species_vocab: Sequence[str] | None = None) -> "LC50Preprocessor":
        if species_vocab is None:
            species_vocab = sorted(frame["Species"].astype(str).unique().tolist())
        species_to_id = {str(s): i for i, s in enumerate(species_vocab)}
        duration = pd.to_numeric(frame["exposure_days"], errors="coerce").to_numpy(dtype=float)
        duration_log = np.log10(np.clip(duration, 1e-12, None))
        dmean = float(np.nanmean(duration_log))
        dstd = float(np.nanstd(duration_log)) or 1.0
        y = pd.to_numeric(frame["log10_lc50_mg_l"], errors="coerce").to_numpy(dtype=float)
        ymean = float(np.nanmean(y))
        ystd = float(np.nanstd(y)) or 1.0
        return cls(species_to_id, dmean, dstd, ymean, ystd)

    @property
    def species_vocab_size(self) -> int:
        return max(len(self.species_to_id), 1)

    def species_id(self, values: Sequence[str]) -> np.ndarray:
       
        return np.asarray([self.species_to_id.get(str(v), 0) for v in values], dtype=np.int64)

    def duration_z(self, values: Sequence[float]) -> np.ndarray:
        x = np.asarray(values, dtype=float)
        x = np.log10(np.clip(x, 1e-12, None))
        return ((x - self.duration_log_mean) / max(self.duration_log_std, 1e-8)).astype(np.float32)

    def target_z(self, values: Sequence[float]) -> np.ndarray:
        x = np.asarray(values, dtype=float)
        return ((x - self.target_mean) / max(self.target_std, 1e-8)).astype(np.float32)

    def inverse_target(self, values: Sequence[float]) -> np.ndarray:
        x = np.asarray(values, dtype=float)
        return x * self.target_std + self.target_mean


def load_and_curate_lc50_fish(path: str | Path) -> LC50CurationResult:
    
    path = Path(path)
    raw = pd.read_excel(path)
    required = [LC50_SPECIES_COL, LC50_DURATION_COL, LC50_TARGET_COL, LC50_SMILES_COL]
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise ValueError(f"LC50 file missing required columns: {missing}")

    work = raw.copy()
    n_input = len(work)
    work["Species"] = work[LC50_SPECIES_COL].astype("string").str.strip()
    work["exposure_days"] = pd.to_numeric(work[LC50_DURATION_COL], errors="coerce")
    work["log10_lc50_mg_l"] = pd.to_numeric(work[LC50_TARGET_COL], errors="coerce")

   
    smi_map: Dict[str, Tuple[str | None, str | None]] = {}
    for smi in work[LC50_SMILES_COL].dropna().astype(str).unique():
        parent, full, _salt, err = standardize_smiles(smi)
        smi_map[str(smi)] = (parent if err is None else None, err)
    work["ParentSMILES"] = [smi_map.get(str(s), (None, "missing"))[0] if pd.notna(s) else None for s in work[LC50_SMILES_COL]]
    work["StructureError"] = [smi_map.get(str(s), (None, "missing"))[1] if pd.notna(s) else "missing" for s in work[LC50_SMILES_COL]]

    valid = (
        work["ParentSMILES"].notna()
        & work["Species"].notna()
        & np.isfinite(work["exposure_days"])
        & (work["exposure_days"] > 0)
        & np.isfinite(work["log10_lc50_mg_l"])
    )
    work = work.loc[valid].copy()

    
    agg_cols = ["ParentSMILES", "Species", "exposure_days"]
    keep_meta = [c for c in [LC50_CAS_COL, "Superclass", "Class", "Phylum", LC50_SMILES_COL, "MolWeight"] if c in work.columns]
    agg = {"log10_lc50_mg_l": "mean"}
    for c in keep_meta:
        agg[c] = "first"
    curated = work.groupby(agg_cols, as_index=False, dropna=False).agg(agg)

    audit = pd.DataFrame([
        {
            "source_file": str(path),
            "input_records": int(n_input),
            "valid_records_before_poststandardization_aggregation": int(len(work)),
            "final_records": int(len(curated)),
            "unique_species": int(curated["Species"].nunique()),
            "unique_parent_structures": int(curated["ParentSMILES"].nunique()),
            "duration_unit": "day",
            "target_unit": "log10(mg/L)",
            "target_already_log10": True,
            "invalid_or_missing_records_removed": int(n_input - len(work)),
            "records_merged_after_standardization": int(len(work) - len(curated)),
        }
    ])
    return LC50CurationResult(curated.reset_index(drop=True), audit)


class LC50Dataset(Dataset):
    def __init__(self, frame: pd.DataFrame, prep: LC50Preprocessor):
        self.frame = frame.reset_index(drop=True)
        self.species = prep.species_id(self.frame["Species"].astype(str).tolist())
        self.duration = prep.duration_z(self.frame["exposure_days"].to_numpy(dtype=float))
        self.y = prep.target_z(self.frame["log10_lc50_mg_l"].to_numpy(dtype=float))

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int) -> dict:
        row = self.frame.iloc[idx]
        return {
            "graph": smiles_to_graph(str(row["ParentSMILES"])),
            "species_id": int(self.species[idx]),
            "duration_z": float(self.duration[idx]),
            "y_z": float(self.y[idx]),
        }


@dataclass
class LC50Batch:
    node_features: torch.Tensor
    edge_index: torch.Tensor
    edge_type: torch.Tensor
    node_batch: torch.Tensor
    species_id: torch.Tensor
    duration_z: torch.Tensor
    y_z: torch.Tensor

    @property
    def batch_size(self) -> int:
        return int(self.y_z.shape[0])

    def to(self, device: torch.device | str) -> "LC50Batch":
        return LC50Batch(
            self.node_features.to(device), self.edge_index.to(device), self.edge_type.to(device),
            self.node_batch.to(device), self.species_id.to(device), self.duration_z.to(device), self.y_z.to(device)
        )


def collate_lc50(records: Sequence[dict]) -> LC50Batch:
    nodes: List[torch.Tensor] = []
    edges: List[torch.Tensor] = []
    types: List[torch.Tensor] = []
    batches: List[torch.Tensor] = []
    species: List[int] = []
    duration: List[float] = []
    ys: List[float] = []
    offset = 0
    for b, rec in enumerate(records):
        g: MoleculeGraph = rec["graph"]
        n = int(g.node_features.shape[0])
        nodes.append(g.node_features)
        batches.append(torch.full((n,), b, dtype=torch.long))
        if g.edge_index.numel() > 0:
            edges.append(g.edge_index + offset)
            types.append(g.edge_type)
        offset += n
        species.append(int(rec["species_id"]))
        duration.append(float(rec["duration_z"]))
        ys.append(float(rec["y_z"]))
    return LC50Batch(
        node_features=torch.cat(nodes, dim=0),
        edge_index=torch.cat(edges, dim=1) if edges else torch.empty((2, 0), dtype=torch.long),
        edge_type=torch.cat(types, dim=0) if types else torch.empty((0,), dtype=torch.long),
        node_batch=torch.cat(batches, dim=0),
        species_id=torch.tensor(species, dtype=torch.long),
        duration_z=torch.tensor(duration, dtype=torch.float32),
        y_z=torch.tensor(ys, dtype=torch.float32),
    )


def make_lc50_loader(
    frame: pd.DataFrame,
    prep: LC50Preprocessor,
    batch_size: int,
    shuffle: bool,
    seed: int = 42,
    num_workers: int = 0,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        LC50Dataset(frame, prep), batch_size=int(batch_size), shuffle=bool(shuffle),
        num_workers=int(num_workers), collate_fn=collate_lc50, drop_last=False,
        generator=generator if shuffle else None,
    )


def _metrics_raw(model: nn.Module, loader: DataLoader, prep: LC50Preprocessor, device: torch.device) -> Dict[str, float]:
    model.eval()
    pred_z: List[float] = []
    true_z: List[float] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred_z.extend(model(batch).detach().cpu().numpy().reshape(-1).tolist())
            true_z.extend(batch.y_z.detach().cpu().numpy().reshape(-1).tolist())
    pred = prep.inverse_target(pred_z)
    true = prep.inverse_target(true_z)
    err = pred - true
    mse = float(np.mean(err ** 2)) if len(err) else float("nan")
    rmse = math.sqrt(mse) if np.isfinite(mse) else float("nan")
    mae = float(np.mean(np.abs(err))) if len(err) else float("nan")
    denom = float(np.sum((true - np.mean(true)) ** 2)) if len(true) else 0.0
    r2 = 1.0 - float(np.sum(err ** 2)) / denom if denom > 0 else float("nan")
    return {"R2": r2, "RMSE": rmse, "MAE": mae, "n": int(len(true))}


def _train_one_lc50_epoch(model, loader, optimizer, device) -> float:
    model.train()
    vals = []
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch)
        loss = F.smooth_l1_loss(pred, batch.y_z, beta=1.0)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        vals.append(float(loss.detach().cpu()))
    return float(np.mean(vals)) if vals else float("nan")


def pretrain_lc50_backbone(
    frame: pd.DataFrame,
    output_dir: str | Path,
    device: torch.device,
    hidden_dim: int = 64,
    n_layers: int = 5,
    dropout: float = 0.10,
    batch_size: int = 128,
    max_epochs: int = 40,
    patience: int = 6,
    seed: int = 42,
    num_workers: int = 0,
) -> Tuple[Dict[str, Dict[str, torch.Tensor]], LC50Preprocessor, pd.DataFrame, int]:
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(seed))
    parents = np.asarray(sorted(frame["ParentSMILES"].astype(str).unique()))
    rng.shuffle(parents)
    n_valid = max(1, int(round(0.10 * len(parents))))
    valid_parents = set(parents[:n_valid].tolist())
    tr = frame.loc[~frame["ParentSMILES"].astype(str).isin(valid_parents)].copy()
    va = frame.loc[frame["ParentSMILES"].astype(str).isin(valid_parents)].copy()
    species_vocab = sorted(frame["Species"].astype(str).unique().tolist())
    prep = LC50Preprocessor.fit(tr, species_vocab=species_vocab)
    train_loader = make_lc50_loader(tr, prep, batch_size, True, seed, num_workers)
    valid_loader = make_lc50_loader(va, prep, batch_size, False, seed, num_workers)

    torch.manual_seed(int(seed))
    model = LC50PretrainRegressor(hidden_dim, n_layers, dropout, prep.species_vocab_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2, min_lr=2e-6)
    history = []
    best_state = None
    best_rmse = float("inf")
    best_epoch = 1
    counter = 0
    for epoch in range(1, int(max_epochs) + 1):
        loss = _train_one_lc50_epoch(model, train_loader, optimizer, device)
        met = _metrics_raw(model, valid_loader, prep, device)
        scheduler.step(met["RMSE"])
        history.append({"stage": "chemical_disjoint_selection", "epoch": epoch, "train_scaled_huber": loss, **met, "lr": optimizer.param_groups[0]["lr"]})
        if met["RMSE"] < best_rmse - 1e-4:
            best_rmse = met["RMSE"]
            best_epoch = epoch
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            counter = 0
        else:
            counter += 1
        if epoch >= 8 and counter >= int(patience):
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    selection_metrics = _metrics_raw(model, valid_loader, prep, device)

    
    full_prep = LC50Preprocessor.fit(frame, species_vocab=species_vocab)
    full_loader = make_lc50_loader(frame, full_prep, batch_size, True, seed + 17, num_workers)
    torch.manual_seed(int(seed) + 17)
    final_model = LC50PretrainRegressor(hidden_dim, n_layers, dropout, full_prep.species_vocab_size).to(device)
    final_opt = torch.optim.AdamW(final_model.parameters(), lr=3e-4, weight_decay=1e-5)
    for epoch in range(1, max(1, int(best_epoch)) + 1):
        loss = _train_one_lc50_epoch(final_model, full_loader, final_opt, device)
        history.append({"stage": "full_refit", "epoch": epoch, "train_scaled_huber": loss, "R2": np.nan, "RMSE": np.nan, "MAE": np.nan, "n": len(frame), "lr": final_opt.param_groups[0]["lr"]})

    transfer_modules = [
        "node_input", "graph_encoder", "pool", "lc50_species_embedding",
        "lc50_time_encoder", "lc50_trunk", "lc50_head",
    ]
    state = {
        name: {k: v.detach().cpu().clone() for k, v in getattr(final_model, name).state_dict().items()}
        for name in transfer_modules
    }
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(output_dir / "lc50_pretraining_history.csv", index=False)
    pd.DataFrame([{**selection_metrics, "best_epoch": int(best_epoch), "n_train_records": len(tr), "n_valid_records": len(va), "n_train_parents": tr["ParentSMILES"].nunique(), "n_valid_parents": va["ParentSMILES"].nunique()}]).to_csv(
        output_dir / "lc50_pretraining_validation_metrics.csv", index=False
    )
    torch.save({"module_states": state, "best_epoch": int(best_epoch), "species_to_id": full_prep.species_to_id,
                "duration_log_mean": full_prep.duration_log_mean, "duration_log_std": full_prep.duration_log_std,
                "target_mean": full_prep.target_mean, "target_std": full_prep.target_std},
               output_dir / "lc50_pretrained_backbone.pt")
    return state, full_prep, hist_df, int(best_epoch)



def load_lc50_pretrained_checkpoint(path: str | Path) -> Tuple[Dict[str, Dict[str, torch.Tensor]], LC50Preprocessor, int]:
    
    payload = torch.load(Path(path), map_location="cpu")
    prep = LC50Preprocessor(
        species_to_id={str(k): int(v) for k, v in payload["species_to_id"].items()},
        duration_log_mean=float(payload["duration_log_mean"]),
        duration_log_std=float(payload["duration_log_std"]),
        target_mean=float(payload["target_mean"]),
        target_std=float(payload["target_std"]),
    )
    return payload["module_states"], prep, int(payload.get("best_epoch", 1))

def load_lc50_state_into_multitask(model: nn.Module, module_states: Dict[str, Dict[str, torch.Tensor]]) -> None:
   
    mapping = {
        "node_input": "lc50_expert_node_input",
        "graph_encoder": "lc50_expert_graph_encoder",
        "pool": "lc50_expert_pool",
    }
    for source_name, target_name in mapping.items():
        state = module_states.get(source_name)
        if state is None or not hasattr(model, target_name):
            continue
        getattr(model, target_name).load_state_dict(state, strict=True)
        for param in getattr(model, target_name).parameters():
            param.requires_grad = False


def train_lc50_aux_steps(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_weight: float,
    max_steps: int = 4,
) -> float:
    
    weight = max(float(loss_weight), 0.0)
    if weight <= 0.0 or max_steps <= 0:
        return float("nan")
    model.train()
    vals: List[float] = []
    for step, batch in enumerate(loader):
        if step >= int(max_steps):
            break
        batch = batch.to(device)
        pred = model.forward_lc50(batch)
        base = F.smooth_l1_loss(pred, batch.y_z, beta=1.0)
        loss = weight * base
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        vals.append(float(base.detach().cpu()))
    return float(np.mean(vals)) if vals else float("nan")
