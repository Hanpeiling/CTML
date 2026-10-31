from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch
from rdkit import Chem


ATOM_SYMBOLS = ["B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Br", "I", "Se", "other"]
HYBRIDIZATIONS = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]


GRAPH_CACHE_VERSION = "rgcn_graph_v2_atom41_rel64"
_GRAPH_CACHE_DIR: Path | None = None
_GRAPH_CACHE_STATS = {"disk_hits": 0, "built": 0, "disk_write_failures": 0}


def _one_hot(value, allowable: Sequence) -> List[float]:
    if value not in allowable:
        return [0.0] * len(allowable) + [1.0]
    return [float(value == x) for x in allowable] + [0.0]


def atom_features(atom: Chem.Atom) -> np.ndarray:
    feats: List[float] = []
    symbol = atom.GetSymbol() if atom.GetSymbol() in ATOM_SYMBOLS[:-1] else "other"
    feats.extend([float(symbol == s) for s in ATOM_SYMBOLS])
    feats.extend(_one_hot(atom.GetDegree(), [0, 1, 2, 3, 4, 5]))
    feats.append(float(atom.GetFormalCharge()))
    feats.append(float(atom.GetNumRadicalElectrons()))
    feats.extend(_one_hot(atom.GetHybridization(), HYBRIDIZATIONS))
    feats.append(float(atom.GetIsAromatic()))
    feats.append(float(atom.IsInRing()))
    feats.extend(_one_hot(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]))
    feats.append(float(atom.GetMass()) / 100.0)
    cip = atom.GetProp("_CIPCode") if atom.HasProp("_CIPCode") else "none"
    feats.extend([float(cip == "R"), float(cip == "S"), float(cip == "none")])
    return np.asarray(feats, dtype=np.float32)


def bond_relation_id(bond: Chem.Bond) -> int:
    """64-way relation id: bond type x conjugation x ring x stereo."""
    bt = bond.GetBondType()
    if bt == Chem.rdchem.BondType.SINGLE:
        base = 0
    elif bt == Chem.rdchem.BondType.DOUBLE:
        base = 1
    elif bt == Chem.rdchem.BondType.TRIPLE:
        base = 2
    else:
        base = 3
    conjugated = 1 if bond.GetIsConjugated() else 0
    ring = 1 if bond.IsInRing() else 0
    stereo_map = {
        Chem.rdchem.BondStereo.STEREONONE: 0,
        Chem.rdchem.BondStereo.STEREOANY: 1,
        Chem.rdchem.BondStereo.STEREOZ: 2,
        Chem.rdchem.BondStereo.STEREOE: 3,
    }
    stereo = stereo_map.get(bond.GetStereo(), 0)
    return int(base + 4 * conjugated + 8 * ring + 16 * stereo)


@dataclass(frozen=True)
class MoleculeGraph:
    node_features: torch.Tensor
    edge_index: torch.Tensor
    edge_type: torch.Tensor


def configure_graph_cache(cache_dir: str | Path | None) -> Path | None:
    """Configure an optional persistent on-disk molecular-graph cache.

    The existing in-memory LRU cache is retained. The persistent cache makes
    repeated HPO/CV/model runs reuse deterministic graph tensors instead of
    reparsing the same standardized SMILES with RDKit each time.
    """
    global _GRAPH_CACHE_DIR
    if cache_dir is None:
        _GRAPH_CACHE_DIR = None
    else:
        root = Path(cache_dir).expanduser().resolve() / GRAPH_CACHE_VERSION
        root.mkdir(parents=True, exist_ok=True)
        _GRAPH_CACHE_DIR = root
  
    smiles_to_graph.cache_clear()
    return _GRAPH_CACHE_DIR


def _cache_path(smiles: str) -> Path | None:
    if _GRAPH_CACHE_DIR is None:
        return None
    digest = hashlib.sha1(smiles.encode("utf-8")).hexdigest()
    return _GRAPH_CACHE_DIR / f"{digest}.pt"


def _graph_from_payload(payload: dict) -> MoleculeGraph:
    return MoleculeGraph(
        node_features=payload["node_features"].to(dtype=torch.float32, device="cpu"),
        edge_index=payload["edge_index"].to(dtype=torch.long, device="cpu"),
        edge_type=payload["edge_type"].to(dtype=torch.long, device="cpu"),
    )


def _build_graph(smiles: str) -> MoleculeGraph:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid standardized SMILES: {smiles}")
    nodes = np.stack([atom_features(a) for a in mol.GetAtoms()], axis=0)
    src: List[int] = []
    dst: List[int] = []
    rel: List[int] = []
    for bond in mol.GetBonds():
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        r = bond_relation_id(bond)
        src.extend([u, v])
        dst.extend([v, u])
        rel.extend([r, r])
    edge_index = (
        torch.tensor([src, dst], dtype=torch.long)
        if src
        else torch.empty((2, 0), dtype=torch.long)
    )
    edge_type = (
        torch.tensor(rel, dtype=torch.long)
        if rel
        else torch.empty((0,), dtype=torch.long)
    )
    return MoleculeGraph(
        torch.tensor(nodes, dtype=torch.float32), edge_index, edge_type
    )


@lru_cache(maxsize=10000)
def smiles_to_graph(smiles: str) -> MoleculeGraph:
    """Convert standardized parent SMILES to a graph with memory+disk caching."""
    smi = str(smiles)
    path = _cache_path(smi)

    if path is not None and path.exists():
        try:
            payload = torch.load(path, map_location="cpu")
            graph = _graph_from_payload(payload)
            _GRAPH_CACHE_STATS["disk_hits"] += 1
            return graph
        except Exception:
        
            try:
                path.unlink(missing_ok=True)
            except Exception:
                pass

    graph = _build_graph(smi)
    _GRAPH_CACHE_STATS["built"] += 1

    if path is not None:
        payload = {
            "node_features": graph.node_features,
            "edge_index": graph.edge_index,
            "edge_type": graph.edge_type,
        }
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        try:
            torch.save(payload, tmp)
            os.replace(tmp, path)
        except Exception:
            _GRAPH_CACHE_STATS["disk_write_failures"] += 1
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass
    return graph


def warm_graph_cache(
    smiles_values: Iterable[str],
    cache_dir: str | Path | None = None,
) -> dict:
    """Eagerly prepare all unique molecular graphs once before HPO/CV.
    """
    if cache_dir is not None:
        configure_graph_cache(cache_dir)

    unique = sorted({str(s) for s in smiles_values if str(s).strip()})
    before = dict(_GRAPH_CACHE_STATS)
    before_info = smiles_to_graph.cache_info()
    for smi in unique:
        smiles_to_graph(smi)
    after_info = smiles_to_graph.cache_info()
    return {
        "n_unique_parent_structures": len(unique),
        "persistent_cache_dir": str(_GRAPH_CACHE_DIR) if _GRAPH_CACHE_DIR else "disabled",
        "disk_hits_this_warmup": _GRAPH_CACHE_STATS["disk_hits"] - before["disk_hits"],
        "graphs_built_this_warmup": _GRAPH_CACHE_STATS["built"] - before["built"],
        "disk_write_failures_this_warmup": _GRAPH_CACHE_STATS["disk_write_failures"] - before["disk_write_failures"],
        "memory_cache_hits_this_warmup": after_info.hits - before_info.hits,
        "memory_cache_size": after_info.currsize,
    }


ATOM_FEATURE_DIM = int(atom_features(Chem.MolFromSmiles("C").GetAtomWithIdx(0)).shape[0])
NUM_RELATIONS = 64


@dataclass
class GraphBatch:
    node_features: torch.Tensor
    edge_index: torch.Tensor
    edge_type: torch.Tensor
    node_batch: torch.Tensor
    context: Dict[str, torch.Tensor]
    y: torch.Tensor
    task_id: torch.Tensor
    row_index: torch.Tensor

    def to(self, device: torch.device | str) -> "GraphBatch":
        return GraphBatch(
            node_features=self.node_features.to(device),
            edge_index=self.edge_index.to(device),
            edge_type=self.edge_type.to(device),
            node_batch=self.node_batch.to(device),
            context={k: v.to(device) for k, v in self.context.items()},
            y=self.y.to(device),
            task_id=self.task_id.to(device),
            row_index=self.row_index.to(device),
        )

    @property
    def batch_size(self) -> int:
        return int(self.y.shape[0])


def collate_graph_records(records: Sequence[dict]) -> GraphBatch:
    node_parts: List[torch.Tensor] = []
    edge_parts: List[torch.Tensor] = []
    type_parts: List[torch.Tensor] = []
    batch_parts: List[torch.Tensor] = []
    
    context_keys = list(records[0]["context"].keys()) if records else []
    contexts: Dict[str, List[torch.Tensor]] = {k: [] for k in context_keys}
    ys: List[float] = []
    task_ids: List[int] = []
    row_indices: List[int] = []
    offset = 0
    for b, rec in enumerate(records):
        g: MoleculeGraph = rec["graph"]
        n = g.node_features.shape[0]
        node_parts.append(g.node_features)
        batch_parts.append(torch.full((n,), b, dtype=torch.long))
        if g.edge_index.numel() > 0:
            edge_parts.append(g.edge_index + offset)
            type_parts.append(g.edge_type)
        offset += n
        for key in contexts:
            contexts[key].append(torch.as_tensor(rec["context"][key], dtype=torch.float32))
        ys.append(float(rec["y"]))
        task_ids.append(int(rec["task_id"]))
        row_indices.append(int(rec["row_index"]))

    return GraphBatch(
        node_features=torch.cat(node_parts, dim=0),
        edge_index=(
            torch.cat(edge_parts, dim=1)
            if edge_parts else torch.empty((2, 0), dtype=torch.long)
        ),
        edge_type=(
            torch.cat(type_parts, dim=0)
            if type_parts else torch.empty((0,), dtype=torch.long)
        ),
        node_batch=torch.cat(batch_parts, dim=0),
        context={k: torch.stack(v, dim=0) for k, v in contexts.items()},
        y=torch.tensor(ys, dtype=torch.float32),
        task_id=torch.tensor(task_ids, dtype=torch.long),
        row_index=torch.tensor(row_indices, dtype=torch.long),
    )


def segment_softmax(scores: torch.Tensor, batch: torch.Tensor, n_graphs: int) -> torch.Tensor:
    max_per = scores.new_full((n_graphs,), -torch.inf)
    max_per.scatter_reduce_(0, batch, scores, reduce="amax", include_self=True)
    exp = torch.exp(scores - max_per[batch])
    denom = scores.new_zeros((n_graphs,))
    denom.index_add_(0, batch, exp)
    return exp / denom[batch].clamp_min(1e-12)


def segment_sum(values: torch.Tensor, batch: torch.Tensor, n_graphs: int) -> torch.Tensor:
    out = values.new_zeros((n_graphs, values.shape[-1]))
    out.index_add_(0, batch, values)
    return out
