from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

EFFECT_SHEETS: Tuple[str, ...] = (
    "Biochemistry",
    "Development",
    "Genetics",
    "Growth",
    "Histology",
    "Hormones",
    "Morphology",
    "Reproduction",
)

TASK_TO_ID: Dict[str, int] = {name.lower(): i for i, name in enumerate(EFFECT_SHEETS)}
ID_TO_TASK: Dict[int, str] = {i: task for task, i in TASK_TO_ID.items()}


SMILES_COL = "SMILES"
CHEMICAL_COL = "Chemicals"
TARGET_CONC_COL = "dose"
TARGET_LOG_COL = "log10_effect_concentration"
GROUP_COL = "ParentSMILES"
TASK_COL = "Task"

FEATURE_GROUPS: Dict[str, List[str]] = {
    "toxicokinetic": ["Fcard", "Cl", "CPBTK"],
    "species_continuous": ["Body weight,g"],
    "species_categorical": ["Species", "Family"],
    "environment_categorical": ["Environment"],
    "endpoint_continuous": ["Exposure time,day"],
    "endpoint_categorical": ["Response Site", "Effect Measurement", "Endpoint"],
}


COVERAGE_COLUMNS: Tuple[str, ...] = (
    "Species",
    "Family",
    "Environment",
    "Response Site",
    "Effect Measurement",
    "Endpoint",
)


DUPLICATE_KEY_COLUMNS: Tuple[str, ...] = (
    TASK_COL,
    GROUP_COL,
    "Fcard",
    "Cl",
    "CPBTK",
    "Species",
    "Body weight,g",
    "Family",
    "Environment",
    "Response Site",
    "Exposure time,day",
    "Effect Measurement",
    "Endpoint",
    TARGET_CONC_COL,
)

DEFAULT_SPLIT_RATIOS = (0.80, 0.10, 0.10)  
DEFAULT_RANDOM_STATE = 42


@dataclass(frozen=True)
class TrainingDefaults:
    max_epochs: int = 250
    patience: int = 30
    hpo_trials: int = 30
    hpo_folds: int = 3
    cv_folds: int = 10
    cv_repeats: int = 1
    batch_size: int = 32
    num_workers: int = 0


DEFAULTS = TrainingDefaults()
