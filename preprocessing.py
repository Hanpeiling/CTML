from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from functools import lru_cache
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.MolStandardize import rdMolStandardize
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import HashingVectorizer

RDLogger.DisableLog("rdApp.info")
RDLogger.DisableLog("rdApp.warning")

from config import (
    CHEMICAL_COL,
    COVERAGE_COLUMNS,
    DEFAULT_RANDOM_STATE,
    DEFAULT_SPLIT_RATIOS,
    DUPLICATE_KEY_COLUMNS,
    EFFECT_SHEETS,
    FEATURE_GROUPS,
    GROUP_COL,
    SMILES_COL,
    TARGET_CONC_COL,
    TARGET_LOG_COL,
    TASK_COL,
)


COLUMN_ALIASES: Mapping[str, Sequence[str]] = {
    "Family": ("Family", "family"),
    "Environment": ("Environment", "environment", "Env", "env"),
    "SMILES": ("SMILES", "smiles", "Smiles"),
    "Cl": ("Cl", "CL", "cl"),
    "CPBTK": ("CPBTK", "Css", "C", "cpbtk"),
    "Fcard": ("Fcard", "FCard", "fcard"),
    "Body weight,g": ("Body weight,g", "Body weight", "Weight", "weight"),
    "Exposure time,day": ("Exposure time,day", "Exposure time", "Time", "time"),
    "Response Site": ("Response Site", "Target", "target"),
    "Effect Measurement": ("Effect Measurement", "Effect measurement", "Indicator", "indicator"),
    "Endpoint": ("Endpoint", "endpoint"),
    "Species": ("Species", "species"),
    "Chemicals": ("Chemicals", "Chemical", "CAS", "CASRN"),
    "dose": ("dose", "Dose", "concentration", "Concentration"),
    "Citation": ("Citation", "citation"),
}


@dataclass
class CurationResult:
    data: pd.DataFrame
    audit: pd.DataFrame
    duplicate_rows: pd.DataFrame
    invalid_smiles_rows: pd.DataFrame


@dataclass
class SplitResult:
    data: pd.DataFrame
    split_summary: pd.DataFrame
    coverage_report: pd.DataFrame
    split_score: float


def _rename_aliases(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    lower = {str(c).strip().lower(): c for c in out.columns}
    rename = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        if canonical in out.columns:
            continue
        for alias in aliases:
            hit = lower.get(alias.strip().lower())
            if hit is not None:
                rename[hit] = canonical
                break
    return out.rename(columns=rename)


def _string_clean(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})


def standardize_smiles(smiles: object) -> Tuple[Optional[str], Optional[str], bool, Optional[str]]:
    
    if smiles is None or (isinstance(smiles, float) and math.isnan(smiles)):
        return None, None, False, "missing"
    raw = str(smiles).strip()
    if not raw:
        return None, None, False, "missing"
    mol = Chem.MolFromSmiles(raw)
    if mol is None:
        return None, None, "." in raw, "invalid_smiles"
    full = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    try:
        
        parent = rdMolStandardize.ChargeParent(mol)
        parent_smiles = Chem.MolToSmiles(parent, canonical=True, isomericSmiles=True)
    except Exception as exc:
        return None, full, "." in full, f"standardization_failed:{exc}"
    return parent_smiles, full, "." in full, None


def _build_cross_sheet_smiles_map(raw_sheets: Mapping[str, pd.DataFrame]) -> Tuple[Dict[str, str], pd.DataFrame]:
    mapping: Dict[str, str] = {}
    rows = []
    for sheet, raw in raw_sheets.items():
        df = _rename_aliases(raw)
        if CHEMICAL_COL not in df.columns or SMILES_COL not in df.columns:
            continue
        for cas, smi in zip(df[CHEMICAL_COL], df[SMILES_COL]):
            if pd.isna(cas) or pd.isna(smi):
                continue
            cas_s = str(cas).strip()
            smi_s = str(smi).strip()
            if not cas_s or not smi_s:
                continue
            if cas_s in mapping and mapping[cas_s] != smi_s:
                rows.append({"CAS": cas_s, "SMILES_A": mapping[cas_s], "SMILES_B": smi_s, "sheet": sheet})
            else:
                mapping[cas_s] = smi_s
    return mapping, pd.DataFrame(rows)


def load_and_curate_workbook(
    workbook_path: str | Path,
    drop_exact_duplicates: bool = True,
    standardize_structures: bool = True,
) -> CurationResult:

    workbook_path = Path(workbook_path)
    xls = pd.ExcelFile(workbook_path)
    missing_sheets = [s for s in EFFECT_SHEETS if s not in xls.sheet_names]
    if missing_sheets:
        raise ValueError(f"Workbook is missing expected sheets: {missing_sheets}")

    raw_sheets = {s: pd.read_excel(workbook_path, sheet_name=s) for s in EFFECT_SHEETS}
    smiles_map, smiles_conflicts = _build_cross_sheet_smiles_map(raw_sheets)

    curated_parts: List[pd.DataFrame] = []
    audit_rows: List[dict] = []

    required_model_cols = [
        CHEMICAL_COL,
        "Species",
        "Body weight,g",
        "Family",
        "Environment",
        "Response Site",
        "Exposure time,day",
        "Effect Measurement",
        "Endpoint",
        "Fcard",
        "Cl",
        "CPBTK",
        TARGET_CONC_COL,
    ]

    for sheet in EFFECT_SHEETS:
        df = _rename_aliases(raw_sheets[sheet])
        df["_source_sheet"] = sheet
        df["_source_excel_row"] = np.arange(2, len(df) + 2)
        df[TASK_COL] = sheet.lower()

        if SMILES_COL not in df.columns:
            df[SMILES_COL] = pd.NA
        if CHEMICAL_COL in df.columns:
            missing_smiles = df[SMILES_COL].isna() | (df[SMILES_COL].astype(str).str.strip() == "")
            df.loc[missing_smiles, SMILES_COL] = (
                df.loc[missing_smiles, CHEMICAL_COL].astype(str).str.strip().map(smiles_map)
            )

        missing = [c for c in required_model_cols if c not in df.columns]
        if missing:
            raise ValueError(f"{sheet}: missing required modeling columns after alias normalization: {missing}")

        
        for c in [CHEMICAL_COL, "Species", "Family", "Environment", "Response Site", "Effect Measurement", "Endpoint"]:
            df[c] = _string_clean(df[c])
        if "Citation" in df.columns:
            df["Citation"] = _string_clean(df["Citation"])

        df["Endpoint"] = df["Endpoint"].str.upper().replace({"LOEL": "LOEC"})
        rows_raw = len(df)
        df = df[df["Endpoint"].isin(["NOEC", "LOEC"])].copy()
        rows_after_endpoint = len(df)

        for c in ["Fcard", "Cl", "CPBTK", "Body weight,g", "Exposure time,day", TARGET_CONC_COL]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df[df[TARGET_CONC_COL].notna() & (df[TARGET_CONC_COL] > 0)].copy()
        rows_after_target = len(df)
        df[TARGET_LOG_COL] = np.log10(df[TARGET_CONC_COL].astype(float))

        if standardize_structures:
            std = df[SMILES_COL].apply(standardize_smiles)
            df[GROUP_COL] = [x[0] for x in std]
            df["CanonicalSMILES_full"] = [x[1] for x in std]
            df["Salt_or_multifragment"] = [x[2] for x in std]
            df["SMILES_error"] = [x[3] for x in std]
        else:
            can = []
            for smi in df[SMILES_COL]:
                mol = Chem.MolFromSmiles(str(smi)) if pd.notna(smi) else None
                can.append(Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True) if mol else None)
            df[GROUP_COL] = can
            df["CanonicalSMILES_full"] = can
            df["Salt_or_multifragment"] = df[SMILES_COL].astype(str).str.contains(".", regex=False)
            df["SMILES_error"] = np.where(df[GROUP_COL].isna(), "invalid_smiles", None)

        audit_rows.append({
            "sheet": sheet,
            "rows_raw": rows_raw,
            "rows_after_endpoint_filter": rows_after_endpoint,
            "rows_after_positive_dose_filter": rows_after_target,
            "rows_removed_non_NOEC_LOEC_or_bad_dose": rows_raw - rows_after_target,
            "rows_missing_or_invalid_parent_smiles": int(df[GROUP_COL].isna().sum()),
            "salt_or_multifragment_rows": int(df["Salt_or_multifragment"].fillna(False).sum()),
        })
        curated_parts.append(df)

    combined = pd.concat(curated_parts, ignore_index=True, sort=False)

    invalid_smiles_rows = combined[combined[GROUP_COL].isna()].copy()
    combined = combined[combined[GROUP_COL].notna()].copy()

    
    duplicate_rows = pd.DataFrame(columns=combined.columns)
    if drop_exact_duplicates:
        missing_dup_cols = [c for c in DUPLICATE_KEY_COLUMNS if c not in combined.columns]
        if missing_dup_cols:
            raise ValueError(f"Duplicate-key columns missing: {missing_dup_cols}")
        dup_mask = combined.duplicated(subset=list(DUPLICATE_KEY_COLUMNS), keep="first")
        duplicate_rows = combined.loc[dup_mask].copy()
        combined = combined.loc[~dup_mask].copy()

    combined = combined.reset_index(drop=True)
    combined["RecordID"] = [f"R{i+1:06d}" for i in range(len(combined))]

    audit = pd.DataFrame(audit_rows)
    audit_extra = pd.DataFrame([
        {
            "sheet": "ALL",
            "rows_raw": sum(len(v) for v in raw_sheets.values()),
            "rows_after_endpoint_filter": np.nan,
            "rows_removed_non_NOEC_LOEC_or_bad_dose": np.nan,
            "rows_missing_or_invalid_parent_smiles": len(invalid_smiles_rows),
            "salt_or_multifragment_rows": int(combined["Salt_or_multifragment"].fillna(False).sum()),
            "exact_duplicate_rows_removed": len(duplicate_rows),
            "unique_CAS_after_curation": combined[CHEMICAL_COL].nunique(dropna=True),
            "unique_parent_structures_after_curation": combined[GROUP_COL].nunique(dropna=True),
            "cross_sheet_smiles_conflicts_detected": len(smiles_conflicts),
        }
    ])
    audit = pd.concat([audit, audit_extra], ignore_index=True, sort=False)
    return CurationResult(combined, audit, duplicate_rows, invalid_smiles_rows)


def _split_summary(df: pd.DataFrame, split_col: str = "split") -> pd.DataFrame:
    rows = []
    for split, sub in df.groupby(split_col, sort=False):
        rows.append({
            "split": split,
            "n_records": len(sub),
            "fraction_records": len(sub) / max(len(df), 1),
            "n_parent_structures": sub[GROUP_COL].nunique(),
            "fraction_parent_structures": sub[GROUP_COL].nunique() / max(df[GROUP_COL].nunique(), 1),
            "n_CAS": sub[CHEMICAL_COL].nunique(),
            "n_species": sub["Species"].nunique(),
            "n_indicators": sub["Effect Measurement"].nunique(),
            "n_response_sites": sub["Response Site"].nunique(),
            "n_NOEC": int((sub["Endpoint"] == "NOEC").sum()),
            "n_LOEC": int((sub["Endpoint"] == "LOEC").sum()),
            "n_source_studies": sub["Citation"].nunique() if "Citation" in sub.columns else np.nan,
            "y_mean": sub[TARGET_LOG_COL].mean(),
            "y_sd": sub[TARGET_LOG_COL].std(ddof=1),
            "y_min": sub[TARGET_LOG_COL].min(),
            "y_max": sub[TARGET_LOG_COL].max(),
        })
    return pd.DataFrame(rows)


def _coverage_report(df: pd.DataFrame, split_col: str = "split") -> pd.DataFrame:
    train = df[df[split_col] == "train"]
    rows: List[dict] = []
    for col in COVERAGE_COLUMNS:
        train_values = set(train[col].dropna().astype(str))
        for split in ["train", "test", "external"]:
            sub_values = set(df.loc[df[split_col] == split, col].dropna().astype(str))
            unseen = sorted(sub_values - train_values) if split != "train" else []
            rows.append({
                "feature": col,
                "split": split,
                "n_unique": len(sub_values),
                "n_unseen_vs_train": len(unseen),
                "unseen_values": " | ".join(unseen),
            })
    return pd.DataFrame(rows)


def _task_species_coverage_audit(df: pd.DataFrame, split_col: str = "split") -> pd.DataFrame:
   
    if TASK_COL not in df.columns or "Species" not in df.columns:
        return pd.DataFrame()

    development = df[df[split_col] != "external"]
    external = df[df[split_col] == "external"]
    rows: List[dict] = []
    pairs = (
        df[[TASK_COL, "Species"]]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .sort_values([TASK_COL, "Species"])
    )
    for _, pair in pairs.iterrows():
        task = pair[TASK_COL]
        species = pair["Species"]
        full_sub = df[(df[TASK_COL].astype(str) == task) & (df["Species"].astype(str) == species)]
        dev_sub = development[(development[TASK_COL].astype(str) == task) & (development["Species"].astype(str) == species)]
        ext_sub = external[(external[TASK_COL].astype(str) == task) & (external["Species"].astype(str) == species)]
        rows.append({
            "task": task,
            "species": species,
            "n_records_total": len(full_sub),
            "n_parent_structures_total": full_sub[GROUP_COL].nunique(),
            "n_records_development": len(dev_sub),
            "n_parent_structures_development": dev_sub[GROUP_COL].nunique(),
            "n_records_external": len(ext_sub),
            "n_parent_structures_external": ext_sub[GROUP_COL].nunique(),
            "retained_in_development": bool(len(dev_sub) > 0),
        })
    return pd.DataFrame(rows)


def _task_species_retention_penalty(
    full: pd.DataFrame,
    development: pd.DataFrame,
) -> float:
  
    if TASK_COL not in full.columns or "Species" not in full.columns:
        return 0.0

    full_pairs = (
        full[[TASK_COL, "Species"]]
        .dropna()
        .astype(str)
        .drop_duplicates()
    )
    dev_pairs = set(
        map(tuple, development[[TASK_COL, "Species"]].dropna().astype(str).drop_duplicates().to_numpy())
    )
    missing = [tuple(x) for x in full_pairs.to_numpy() if tuple(x) not in dev_pairs]

    
    penalty = 5000.0 * len(missing)

  
    full_counts = full.groupby([TASK_COL, "Species"]).size()
    dev_counts = development.groupby([TASK_COL, "Species"]).size()
    full_parents = full.groupby([TASK_COL, "Species"])[GROUP_COL].nunique()
    dev_parents = development.groupby([TASK_COL, "Species"])[GROUP_COL].nunique()
    for key, n_total in full_counts.items():
        n_total = max(int(n_total), 1)
        n_dev = int(dev_counts.get(key, 0))
        p_total = max(int(full_parents.get(key, 0)), 1)
        p_dev = int(dev_parents.get(key, 0))
        record_retention = n_dev / n_total
        parent_retention = p_dev / p_total
        rarity_weight = 1.0 / math.sqrt(float(p_total))
        penalty += 8.0 * rarity_weight * (1.0 - record_retention)
        penalty += 18.0 * rarity_weight * (1.0 - parent_retention)
    return float(penalty)


def _parent_structure_cluster_map(groups: Sequence[str], random_state: int) -> Dict[str, int]:
    
    groups = list(groups)
    if len(groups) <= 2:
        return {g: i for i, g in enumerate(groups)}
    rows = []
    for smi in groups:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            rows.append(np.zeros(512, dtype=np.float32))
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=512)
        rows.append(np.asarray(list(fp), dtype=np.float32))
    X = np.vstack(rows)
    n_clusters = int(round(math.sqrt(len(groups))))
    n_clusters = max(3, min(10, n_clusters, len(groups)))
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = km.fit_predict(X)
    return {g: int(lbl) for g, lbl in zip(groups, labels)}


def _task_external_candidate_score(
    full_df: pd.DataFrame,
    task_df: pd.DataFrame,
    task: str,
    external_groups: set[str],
    ratios: Tuple[float, float, float],
    structure_clusters: Optional[Dict[str, int]] = None,
    prior_external_use: Optional[Dict[str, int]] = None,
) -> float:
   
    external = task_df[task_df[GROUP_COL].isin(external_groups)]
    development = task_df[~task_df[GROUP_COL].isin(external_groups)]
    if external.empty or development.empty:
        return 1e12

    score = 0.0
    n = max(len(task_df), 1)
    n_groups_total = max(task_df[GROUP_COL].nunique(), 1)
    target_external = float(ratios[2])

   
    row_frac = len(external) / n
    group_frac = external[GROUP_COL].nunique() / n_groups_total
    score += 45.0 * abs(row_frac - target_external)
    score += 130.0 * abs(group_frac - target_external)

   
    full_species = set(task_df["Species"].dropna().astype(str))
    dev_species = set(development["Species"].dropna().astype(str))
    score += 1_000_000.0 * len(full_species - dev_species)

    
    for col in ["Family", "Environment", "Effect Measurement", "Response Site", "Endpoint"]:
        if col not in task_df.columns:
            continue
        full_values = set(task_df[col].dropna().astype(str))
        dev_values = set(development[col].dropna().astype(str))
        score += 10_000.0 * len(full_values - dev_values)

    
    full_counts = task_df.groupby("Species").size()
    dev_counts = development.groupby("Species").size()
    full_parents = task_df.groupby("Species")[GROUP_COL].nunique()
    dev_parents = development.groupby("Species")[GROUP_COL].nunique()
    for species, n_total in full_counts.items():
        n_total = max(int(n_total), 1)
        n_dev = int(dev_counts.get(species, 0))
        p_total = max(int(full_parents.get(species, 0)), 1)
        p_dev = int(dev_parents.get(species, 0))
        rarity = 1.0 / math.sqrt(float(p_total))
        score += 10.0 * rarity * (1.0 - n_dev / n_total)
        score += 22.0 * rarity * (1.0 - p_dev / p_total)

    
    if structure_clusters:
        all_groups = pd.DataFrame({GROUP_COL: task_df[GROUP_COL].dropna().unique()})
        all_groups["_cluster"] = all_groups[GROUP_COL].map(structure_clusters)
        ext_groups_df = pd.DataFrame({GROUP_COL: list(external_groups)})
        ext_groups_df["_cluster"] = ext_groups_df[GROUP_COL].map(structure_clusters)
        global_prop = all_groups["_cluster"].value_counts(normalize=True)
        ext_prop = ext_groups_df["_cluster"].value_counts(normalize=True)
        keys = set(global_prop.index).union(ext_prop.index)
        score += 15.0 * sum(
            abs(float(ext_prop.get(k, 0.0)) - float(global_prop.get(k, 0.0)))
            for k in keys
        )

    
    if "Endpoint" in task_df.columns:
        gp = task_df["Endpoint"].value_counts(normalize=True, dropna=False)
        ep = external["Endpoint"].value_counts(normalize=True, dropna=False)
        keys = set(gp.index).union(ep.index)
        score += 8.0 * sum(abs(float(ep.get(k, 0.0)) - float(gp.get(k, 0.0))) for k in keys)

    mean_all = float(task_df[TARGET_LOG_COL].mean())
    sd_all = float(task_df[TARGET_LOG_COL].std(ddof=0)) or 1.0
    score += 2.5 * abs(float(external[TARGET_LOG_COL].mean()) - mean_all) / sd_all
    ext_sd = float(external[TARGET_LOG_COL].std(ddof=0)) if len(external) > 1 else 0.0
    score += 1.0 * abs(ext_sd - sd_all) / sd_all

    
    other_task_df = full_df.loc[full_df[TASK_COL].astype(str) != str(task)].copy()
    other_task_parents = set(other_task_df[GROUP_COL].dropna().astype(str))
    transferable = sum(1 for g in external_groups if str(g) in other_task_parents)
    transfer_fraction = transferable / max(len(external_groups), 1)

    
    score += 100_000.0 * (1.0 - transfer_fraction)

    
    support_tasks = other_task_df.groupby(GROUP_COL)[TASK_COL].nunique().astype(float).to_dict()
    support_records = other_task_df.groupby(GROUP_COL).size().astype(float).to_dict()
    max_other_tasks = max(int(full_df[TASK_COL].astype(str).nunique()) - 1, 1)
    if external_groups:
        task_support_fraction = np.mean([
            float(support_tasks.get(str(g), 0.0)) / max_other_tasks
            for g in external_groups
        ])
        record_support = np.mean([
            math.log1p(float(support_records.get(str(g), 0.0)))
            for g in external_groups
        ])
        score += 4_000.0 * (1.0 - float(task_support_fraction))
        score -= 25.0 * float(record_support)


    if prior_external_use:
        score += 50_000.0 * sum(int(prior_external_use.get(str(g), 0)) for g in external_groups)

    return float(score)

def _internal_candidate_score(
    development: pd.DataFrame,
    test_indices: set[int],
    target_test_rows: int,
    structure_clusters: Optional[Dict[str, int]] = None,
) -> float:
 
    test = development.loc[development.index.isin(test_indices)]
    train = development.loc[~development.index.isin(test_indices)]
    if train.empty or test.empty:
        return 1e12

    score = 0.0
    score += 25.0 * abs(len(test) - target_test_rows) / max(target_test_rows, 1)

  
    train_groups = set(train[GROUP_COL].astype(str))
    test_groups = set(test[GROUP_COL].astype(str))
    if not test_groups.issubset(train_groups):
        score += 1e8 * len(test_groups - train_groups)

    
    for col in COVERAGE_COLUMNS:
        train_values = set(train[col].dropna().astype(str))
        dev_values = set(development[col].dropna().astype(str))
        score += 600.0 * len(dev_values - train_values)

   
    for col in ['Endpoint', TASK_COL]:
        global_prop = development[col].value_counts(normalize=True, dropna=False)
        test_prop = test[col].value_counts(normalize=True, dropna=False)
        keys = set(global_prop.index).union(test_prop.index)
        score += 8.0 * sum(
            abs(float(test_prop.get(k, 0.0)) - float(global_prop.get(k, 0.0)))
            for k in keys
        )

    global_mean = float(development[TARGET_LOG_COL].mean())
    global_sd = float(development[TARGET_LOG_COL].std(ddof=0)) or 1.0
    score += 2.5 * abs(float(test[TARGET_LOG_COL].mean()) - global_mean) / global_sd
    test_sd = float(test[TARGET_LOG_COL].std(ddof=0)) if len(test) > 1 else 0.0
    score += abs(test_sd - global_sd) / global_sd


    if len(test_groups) > 1:
        parent_prop = test[GROUP_COL].value_counts(normalize=True)
        score += 2.0 * float((parent_prop ** 2).sum())

   
    if structure_clusters:
        train_test_groups = pd.DataFrame({GROUP_COL: list(test_groups)})
        train_test_groups['_cluster'] = train_test_groups[GROUP_COL].map(structure_clusters)
        all_dev_groups = pd.DataFrame({GROUP_COL: development[GROUP_COL].dropna().unique()})
        all_dev_groups['_cluster'] = all_dev_groups[GROUP_COL].map(structure_clusters)
        global_prop = all_dev_groups['_cluster'].value_counts(normalize=True)
        test_prop = train_test_groups['_cluster'].value_counts(normalize=True)
        keys = set(global_prop.index).union(test_prop.index)
        score += 3.0 * sum(
            abs(float(test_prop.get(k, 0.0)) - float(global_prop.get(k, 0.0)))
            for k in keys
        )
    return float(score)


def _record_level_internal_split(
    development: pd.DataFrame,
    target_test_rows: int,
    random_state: int,
    n_trials: int,
    structure_clusters: Optional[Dict[str, int]] = None,
) -> Tuple[set[int], set[int], float]:
   
    if target_test_rows <= 0:
        raise ValueError('target_test_rows must be positive')

    group_to_indices = {
        str(g): list(idx)
        for g, idx in development.groupby(GROUP_COL, sort=False).groups.items()
    }
    capacity = sum(max(0, len(indices) - 1) for indices in group_to_indices.values())
    if capacity < 1:
        raise ValueError('Not enough repeated parent structures to create an internal test set.')
    target_test_rows = min(int(target_test_rows), int(capacity))

    rng = random.Random(random_state)
    all_indices = list(development.index)
    best_test: Optional[set[int]] = None
    best_score = float('inf')

    for _ in range(max(100, min(int(n_trials), 1500))):
        order = all_indices.copy()
        rng.shuffle(order)
        selected: set[int] = set()
        selected_per_group: Dict[str, int] = {g: 0 for g in group_to_indices}
        max_test_per_group = {g: max(0, len(ids) - 1) for g, ids in group_to_indices.items()}

        
        for idx in order:
            if len(selected) >= target_test_rows:
                break
            g = str(development.at[idx, GROUP_COL])
            if selected_per_group[g] < max_test_per_group[g]:
                selected.add(idx)
                selected_per_group[g] += 1

        if len(selected) < target_test_rows:
            continue
        score = _internal_candidate_score(
            development,
            selected,
            target_test_rows,
            structure_clusters=structure_clusters,
        )
        if score < best_score:
            best_score = score
            best_test = selected.copy()

    if best_test is None:
        raise RuntimeError('Could not construct a valid internal train/test split.')
    train_idx = set(development.index) - best_test
    return train_idx, best_test, float(best_score)


def constrained_parent_structure_split(
    df: pd.DataFrame,
    ratios: Tuple[float, float, float] = DEFAULT_SPLIT_RATIOS,
    random_state: int = DEFAULT_RANDOM_STATE,
    n_trials: int = 4000,
) -> SplitResult:
 
    if not math.isclose(sum(ratios), 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("split ratios must sum to 1")
    if TASK_COL not in df.columns:
        raise ValueError(f"Missing task column: {TASK_COL}")

    all_groups = df[GROUP_COL].dropna().astype(str).unique().tolist()
    structure_clusters = _parent_structure_cluster_map(all_groups, random_state=random_state)
    rng = random.Random(random_state)

    
    task_parent_counts = df.groupby(TASK_COL)[GROUP_COL].nunique().sort_values()
    task_order = [str(t) for t in task_parent_counts.index]

    external_pairs: set[tuple[str, str]] = set()
    prior_external_use: Dict[str, int] = {}
    total_score = 0.0

    print(f"[split] Selecting task-specific external chemicals for {len(task_order)} tasks...", flush=True)
    for task_index, task in enumerate(task_order):
        task_df = df[df[TASK_COL].astype(str) == task].copy()
        groups = task_df[GROUP_COL].dropna().astype(str).unique().tolist()
        if len(groups) < 2:
            continue

        n_external = max(1, int(round(float(ratios[2]) * len(groups))))
        n_external = min(n_external, max(1, len(groups) - 1))

        best_groups: Optional[set[str]] = None
        best_score = float("inf")
        print(
            f"[split] Task {task_index + 1}/{len(task_order)} {task}: "
            f"parents={len(groups)}, target_external={n_external}, trials={max(200, min(int(n_trials), 5000))}",
            flush=True,
        )
        local_rng = random.Random(random_state + 1009 * (task_index + 1))
        trials = max(200, min(int(n_trials), 5000))

      
        other_task_groups = set(
            df.loc[df[TASK_COL].astype(str) != str(task), GROUP_COL]
            .dropna().astype(str)
        )
        transferable_unique = [
            g for g in groups
            if str(g) in other_task_groups and prior_external_use.get(str(g), 0) == 0
        ]
        transferable_reused = [
            g for g in groups
            if str(g) in other_task_groups and prior_external_use.get(str(g), 0) > 0
        ]
        nontransferable = [g for g in groups if str(g) not in other_task_groups]

        for _ in range(trials):
            cand_list: List[str] = []
            pool1 = transferable_unique.copy(); local_rng.shuffle(pool1)
            cand_list.extend(pool1[:n_external])

            if len(cand_list) < n_external:
                
                pool2 = transferable_reused.copy(); local_rng.shuffle(pool2)
                need = n_external - len(cand_list)
                cand_list.extend(pool2[:need])

            if len(cand_list) < n_external:
                pool3 = nontransferable.copy(); local_rng.shuffle(pool3)
                need = n_external - len(cand_list)
                cand_list.extend(pool3[:need])

       
            if len(transferable_unique) >= n_external:
                pool1 = transferable_unique.copy(); local_rng.shuffle(pool1)
                cand_list = pool1[:n_external]

            cand = set(cand_list[:n_external])
            score = _task_external_candidate_score(
                df,
                task_df,
                task,
                cand,
                ratios,
                structure_clusters=structure_clusters,
                prior_external_use=prior_external_use,
            )
            if score < best_score:
                best_score = score
                best_groups = cand.copy()

        if best_groups is None:
            raise RuntimeError(f"Could not construct a task-specific external holdout for {task}.")

        other_task_groups_final = set(
            df.loc[df[TASK_COL].astype(str) != str(task), GROUP_COL].dropna().astype(str)
        )
        transfer_n = sum(str(g) in other_task_groups_final for g in best_groups)
        print(
            f"[split] Task {task}: selected {len(best_groups)} external parents; "
            f"cross-task-supported={transfer_n}/{len(best_groups)}; species/context score={best_score:.2f}",
            flush=True,
        )
        for g in best_groups:
            external_pairs.add((task, str(g)))
            prior_external_use[str(g)] = prior_external_use.get(str(g), 0) + 1
        total_score += float(best_score)

    out = df.copy()
    pair_keys = list(zip(out[TASK_COL].astype(str), out[GROUP_COL].astype(str)))
    external_mask = np.array([pair in external_pairs for pair in pair_keys], dtype=bool)
    out["split"] = "train"
    out.loc[external_mask, "split"] = "external"
    out["StructureCluster"] = out[GROUP_COL].map(structure_clusters)

   
    development = out[out["split"] != "external"].copy()
    target_test_rows = max(1, int(round(float(ratios[1]) * len(df))))
    try:
        train_idx, test_idx, internal_score = _record_level_internal_split(
            development,
            target_test_rows=target_test_rows,
            random_state=random_state + 17,
            n_trials=n_trials,
            structure_clusters=structure_clusters,
        )
        out.loc[out.index.isin(test_idx), "split"] = "test"
        total_score += float(internal_score)
    except Exception:
        
        pass

   
    for task in out[TASK_COL].dropna().astype(str).unique():
        task_sub = out[out[TASK_COL].astype(str) == task]
        ext = set(task_sub.loc[task_sub["split"] == "external", GROUP_COL].astype(str))
        dev = set(task_sub.loc[task_sub["split"] != "external", GROUP_COL].astype(str))
        if not ext.isdisjoint(dev):
            raise AssertionError(f"Task-specific external leakage detected for task={task}")

    return SplitResult(
        data=out,
        split_summary=_split_summary(out),
        coverage_report=_coverage_report(out),
        split_score=float(total_score),
    )

@lru_cache(maxsize=10000)
def _cached_ecfp4(smiles: str, n_bits: int = 1024) -> np.ndarray:
    mol = Chem.MolFromSmiles(str(smiles))
    arr = np.zeros((int(n_bits),), dtype=np.int8)
    if mol is not None:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=int(n_bits))
        DataStructs.ConvertToNumpyArray(fp, arr)
    return arr.astype(np.float32)


class ContextPreprocessor:
   

    MEASUREMENT_CHAR_FEATURES = 128
    MEASUREMENT_WORD_FEATURES = 64
    ECFP_BITS = 1024

    def __init__(self, measurement_encoding_mode: str = "hybrid"):
        mode = str(measurement_encoding_mode).strip().lower()
        if mode not in {"hybrid", "exact_only"}:
            raise ValueError("measurement_encoding_mode must be 'hybrid' or 'exact_only'")
        self.measurement_encoding_mode = mode
        self.tk_imputer = SimpleImputer(strategy="median")
        self.tk_scaler = StandardScaler()

        self.spec_cont_imputer = SimpleImputer(strategy="median")
        self.spec_cont_scaler = StandardScaler()
        self.spec_cat_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)

        self.env_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)

        self.end_cont_imputer = SimpleImputer(strategy="median")
        self.end_cont_scaler = StandardScaler()
        
        self.endpoint_exact_cols = ["Response Site", "Endpoint"]
        self.end_cat_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)

        self.measurement_exact_encoder = OneHotEncoder(
            handle_unknown="ignore", sparse_output=False, dtype=np.float32
        )
        self.measurement_char_vectorizer = HashingVectorizer(
            analyzer="char_wb", ngram_range=(3, 5),
            n_features=self.MEASUREMENT_CHAR_FEATURES,
            alternate_sign=False, norm="l2", lowercase=True,
        )
        self.measurement_word_vectorizer = HashingVectorizer(
            analyzer="word", ngram_range=(1, 2),
            n_features=self.MEASUREMENT_WORD_FEATURES,
            alternate_sign=False, norm="l2", lowercase=True,
            token_pattern=r"(?u)\b[\w-]+\b",
        )
        self._fitted = False

    @staticmethod
    def _cat_frame(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
        out = df.loc[:, list(cols)].copy()
        for col in out.columns:
            out[col] = out[col].astype("string").fillna("__MISSING__").astype(str)
        return out

    @staticmethod
    def _num_frame(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
       
        out = df.loc[:, list(cols)].copy()
        for c in out.columns:
            values = pd.to_numeric(out[c], errors="coerce")
            values = values.where(values > 0, np.nan)
            out[c] = np.log10(values)
        return out

    @classmethod
    def _ecfp_features(cls, df: pd.DataFrame) -> np.ndarray:
       
        out = np.zeros((len(df), cls.ECFP_BITS), dtype=np.float32)
        for i, smi in enumerate(df[GROUP_COL].astype(str).tolist()):
            out[i] = _cached_ecfp4(str(smi), cls.ECFP_BITS)
        return out

    @staticmethod
    def _measurement_text(df: pd.DataFrame) -> list[str]:
        if "Effect Measurement" not in df.columns:
            return ["__missing__"] * len(df)
        return (
            df["Effect Measurement"]
            .astype("string")
            .fillna("__MISSING__")
            .astype(str)
            .str.strip()
            .str.lower()
            .tolist()
        )

    def fit(self, df: pd.DataFrame) -> "ContextPreprocessor":
        tk = self._num_frame(df, FEATURE_GROUPS["toxicokinetic"])
        self.tk_scaler.fit(self.tk_imputer.fit_transform(tk))

        sc = self._num_frame(df, FEATURE_GROUPS["species_continuous"])
        self.spec_cont_scaler.fit(self.spec_cont_imputer.fit_transform(sc))
        self.spec_cat_encoder.fit(self._cat_frame(df, FEATURE_GROUPS["species_categorical"]))

        self.env_encoder.fit(self._cat_frame(df, FEATURE_GROUPS["environment_categorical"]))

        ec = self._num_frame(df, FEATURE_GROUPS["endpoint_continuous"])
        self.end_cont_scaler.fit(self.end_cont_imputer.fit_transform(ec))
        self.end_cat_encoder.fit(self._cat_frame(df, self.endpoint_exact_cols))
        self.measurement_exact_encoder.fit(self._cat_frame(df, ["Effect Measurement"]))
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        if not self._fitted:
            raise RuntimeError("ContextPreprocessor must be fit before transform.")
        tk = self.tk_scaler.transform(
            self.tk_imputer.transform(self._num_frame(df, FEATURE_GROUPS["toxicokinetic"]))
        ).astype(np.float32)

        spec_cont = self.spec_cont_scaler.transform(
            self.spec_cont_imputer.transform(self._num_frame(df, FEATURE_GROUPS["species_continuous"]))
        ).astype(np.float32)
        spec_cat = self.spec_cat_encoder.transform(
            self._cat_frame(df, FEATURE_GROUPS["species_categorical"])
        ).astype(np.float32)
        species = np.concatenate([spec_cont, spec_cat], axis=1).astype(np.float32)

        env = self.env_encoder.transform(
            self._cat_frame(df, FEATURE_GROUPS["environment_categorical"])
        ).astype(np.float32)

        end_cont = self.end_cont_scaler.transform(
            self.end_cont_imputer.transform(self._num_frame(df, FEATURE_GROUPS["endpoint_continuous"]))
        ).astype(np.float32)
        end_cat = self.end_cat_encoder.transform(
            self._cat_frame(df, self.endpoint_exact_cols)
        ).astype(np.float32)
        endpoint = np.concatenate([end_cont, end_cat], axis=1).astype(np.float32)

        measurement_exact = self.measurement_exact_encoder.transform(
            self._cat_frame(df, ["Effect Measurement"])
        ).astype(np.float32)
        if self.measurement_encoding_mode == "exact_only":
            measurement = measurement_exact.astype(np.float32)
        else:
            texts = self._measurement_text(df)
            measurement_char = self.measurement_char_vectorizer.transform(texts).toarray().astype(np.float32)
            measurement_word = self.measurement_word_vectorizer.transform(texts).toarray().astype(np.float32)
            measurement = np.concatenate(
                [measurement_exact, measurement_char, measurement_word], axis=1
            ).astype(np.float32)

        return {
            "tk": tk,
            "species": species,
            "environment": env,
            "endpoint": endpoint,
            "measurement": measurement,
            "fingerprint": self._ecfp_features(df),
        }

    def fit_transform(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        return self.fit(df).transform(df)

    @property
    def output_dims(self) -> Dict[str, int]:
        if not self._fitted:
            raise RuntimeError("Preprocessor not fitted")
        return {
            "tk": len(FEATURE_GROUPS["toxicokinetic"]),
            "species": len(FEATURE_GROUPS["species_continuous"]) + int(sum(len(c) for c in self.spec_cat_encoder.categories_)),
            "environment": int(sum(len(c) for c in self.env_encoder.categories_)),
            "endpoint": len(FEATURE_GROUPS["endpoint_continuous"]) + int(sum(len(c) for c in self.end_cat_encoder.categories_)),
            "measurement": (
                int(sum(len(c) for c in self.measurement_exact_encoder.categories_))
                if self.measurement_encoding_mode == "exact_only"
                else int(sum(len(c) for c in self.measurement_exact_encoder.categories_))
                    + self.MEASUREMENT_CHAR_FEATURES + self.MEASUREMENT_WORD_FEATURES
            ),
            "fingerprint": self.ECFP_BITS,
        }

    def save(self, path: str | Path) -> None:
        joblib.dump(self, path)


def write_curation_and_split_outputs(
    output_dir: str | Path,
    curation: CurationResult,
    split: SplitResult,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    curation.audit.to_csv(output_dir / "curation_audit.csv", index=False)
    curation.duplicate_rows.to_csv(output_dir / "removed_exact_duplicates.csv", index=False)
    curation.invalid_smiles_rows.to_csv(output_dir / "excluded_invalid_or_missing_smiles.csv", index=False)
    split.data.to_csv(output_dir / "curated_data_with_splits.csv", index=False)
    split.split_summary.to_csv(output_dir / "split_summary.csv", index=False)
    split.coverage_report.to_csv(output_dir / "categorical_coverage_report.csv", index=False)

    task_species_audit = _task_species_coverage_audit(split.data)
    if not task_species_audit.empty:
        task_species_audit.to_csv(output_dir / "task_species_coverage_audit.csv", index=False)
        summary = (
            task_species_audit.groupby("task", as_index=False)
            .agg(
                n_species_total=("species", "nunique"),
                n_species_retained_in_development=("retained_in_development", "sum"),
            )
        )
        summary["species_coverage_fraction"] = (
            summary["n_species_retained_in_development"] / summary["n_species_total"].clip(lower=1)
        )
        summary.to_csv(output_dir / "task_species_coverage_summary.csv", index=False)

    
    split.data[[TASK_COL, "split", GROUP_COL]].drop_duplicates().sort_values(
        [TASK_COL, "split", GROUP_COL]
    ).to_csv(output_dir / "split_parent_structures_by_task.csv", index=False)

    development = split.data[split.data["split"] != "external"]
    external = split.data[split.data["split"] == "external"]
    audit_rows = []
    for task in sorted(split.data[TASK_COL].dropna().astype(str).unique()):
        task_all = split.data[split.data[TASK_COL].astype(str) == task]
        task_dev = development[development[TASK_COL].astype(str) == task]
        task_ext = external[external[TASK_COL].astype(str) == task]
        ext_parents = set(task_ext[GROUP_COL].astype(str))
        dev_same_task = set(task_dev[GROUP_COL].astype(str))
        other_task_dev_parents = set(
            development.loc[development[TASK_COL].astype(str) != task, GROUP_COL].astype(str)
        )
        seen_elsewhere = ext_parents & other_task_dev_parents
        other_dev = development[development[TASK_COL].astype(str) != task]
        other_task_support = other_dev.groupby(GROUP_COL)[TASK_COL].nunique().astype(int).to_dict()
        support_values = [int(other_task_support.get(str(g), 0)) for g in ext_parents]
        audit_rows.append({
            "task": task,
            "n_records_total": len(task_all),
            "n_records_external": len(task_ext),
            "n_parent_structures_total": task_all[GROUP_COL].nunique(),
            "n_parent_structures_external": len(ext_parents),
            "same_task_external_development_overlap": len(ext_parents & dev_same_task),
            "n_external_parents_seen_in_other_task_development": len(seen_elsewhere),
            "fraction_external_parents_seen_in_other_task_development": (
                len(seen_elsewhere) / max(len(ext_parents), 1)
            ),
            "mean_number_of_other_training_tasks_per_external_parent": (
                float(np.mean(support_values)) if support_values else 0.0
            ),
            "min_number_of_other_training_tasks_per_external_parent": (
                int(min(support_values)) if support_values else 0
            ),
            "fraction_external_parents_seen_in_at_least_2_other_tasks": (
                sum(v >= 2 for v in support_values) / max(len(support_values), 1)
            ),
            "fraction_external_parents_seen_in_at_least_3_other_tasks": (
                sum(v >= 3 for v in support_values) / max(len(support_values), 1)
            ),
            "n_external_parents_globally_unseen": len(ext_parents - other_task_dev_parents),
        })
    pd.DataFrame(audit_rows).to_csv(output_dir / "task_transfer_external_audit.csv", index=False)

    id_cols = [
        c for c in [
            "split", CHEMICAL_COL, GROUP_COL, "Species", "Effect Measurement",
            "Endpoint", "Response Site", TASK_COL
        ] if c in split.data.columns
    ]
    split.data[id_cols].drop_duplicates().sort_values(id_cols).to_csv(
        output_dir / "split_identifiers.csv", index=False
    )

    with open(output_dir / "split_metadata.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "split_score": split.split_score,
                "grouping_key": f"{TASK_COL} × {GROUP_COL}",
                "ratios": DEFAULT_SPLIT_RATIOS,
                "split_design": {
                    "train": "task-specific model-development records",
                    "test": "optional internal record-level diagnostic set; combined with train for final multitask development",
                    "external": (
                        "task-specific chemical holdout: a parent structure is absent from development for the target task; "
                        "selection lexicographically prioritizes target-task external parents that remain in development for other tasks, then maximizes the number of supporting other tasks while preserving target-task species coverage"
                    ),
                },
                "parent_structure_constraints": {
                    "external_disjoint_from_same_task_development": True,
                    "external_may_overlap_other_task_development": True,
                    "external_other_task_overlap_is_preferred": True,
                    "global_external_parent_disjointness_required": False,
                },
                "task_specific_coverage_priority": [
                    "Task × Species",
                    "Task × Effect Measurement",
                    "Task × Response Site",
                    "Task × Endpoint",
                ],
                "validation_interpretation": (
                    "External performance measures task-specific chemical transfer, not globally unseen-chemical extrapolation."
                ),
                "duplicate_key_columns": list(DUPLICATE_KEY_COLUMNS),
            },
            f,
            indent=2,
        )

