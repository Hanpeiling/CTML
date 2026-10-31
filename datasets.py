from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from config import GROUP_COL, TARGET_LOG_COL, TASK_COL, TASK_TO_ID
from graph_utils import smiles_to_graph


class MultimodalGraphDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, context_arrays: Dict[str, np.ndarray]):
        self.frame = frame.reset_index(drop=False).rename(columns={"index": "_original_frame_index"})
        self.context = context_arrays
        n = len(self.frame)
        for key, arr in self.context.items():
            if len(arr) != n:
                raise ValueError(f"Context array {key} length {len(arr)} != frame length {n}")

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int) -> dict:
        row = self.frame.iloc[idx]
        return {
            "graph": smiles_to_graph(str(row[GROUP_COL])),
            "context": {k: v[idx] for k, v in self.context.items()},
            "y": float(row[TARGET_LOG_COL]),
            "task_id": TASK_TO_ID[str(row[TASK_COL]).lower()],
            "row_index": int(row["_original_frame_index"]),
        }
