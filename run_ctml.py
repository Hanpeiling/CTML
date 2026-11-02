from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
import torch

from config import GROUP_COL, TASK_COL, TARGET_LOG_COL, TASK_TO_ID
from multitask_training import run_multitask_training

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "curated_data_with_splits.csv"
PARAM_PATH = ROOT / "fixed_hyperparameters.json"
FROZEN_PATH = ROOT / "weights" / "frozen_lc50_expert.pt"
REFERENCE_PATH = ROOT / "model" / "multitask_model.pt"
MANIFEST_PATH = ROOT / "SHA256SUMS.json"

FIXED_CONFIG = {
    "seed": 42,
    "cv_folds": 10,
    "cv_repeats": 2,
    "max_epochs": 180,
    "patience": 15,
    "final_ensemble_seeds": 3,
    "num_workers": 0,
    "uncertainty_lr_multiplier": 5.0,
    "uncertainty_warmup_epochs": 8,
    "calibration_strength": 50.0,
    "measurement_encoding_mode": "exact_only",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify_manifest() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    for relative, expected in manifest.items():
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"Required release asset missing: {relative}")
        if sha256(path) != expected:
            raise ValueError(f"Release asset checksum mismatch: {relative}")


def verify_data(frame: pd.DataFrame, reference: dict) -> None:
    if len(frame) != 5005:
        raise ValueError(f"Expected 5005 curated records, found {len(frame)}")
    if frame["Species"].nunique() != 67:
        raise ValueError("Expected 67 species")
    if set(frame[TASK_COL].dropna().astype(str).str.lower()) != set(TASK_TO_ID):
        raise ValueError("Unexpected effect-category names")
    if frame[[TASK_COL, "Effect Measurement"]].drop_duplicates().shape[0] != 108:
        raise ValueError("Expected 108 task-specific effect measurements")
    if frame["split"].value_counts().to_dict() != {"train": 4177, "test": 500, "external": 328}:
        raise ValueError("Training/test/external labels do not match the release")
    if not pd.to_numeric(frame[TARGET_LOG_COL], errors="coerce").notna().all():
        raise ValueError("Invalid log10 target values")
    for task in TASK_TO_ID:
        task_rows = frame[frame[TASK_COL].str.lower() == task]
        dev = set(task_rows.loc[task_rows["split"] != "external", GROUP_COL].astype(str))
        ext = set(task_rows.loc[task_rows["split"] == "external", GROUP_COL].astype(str))
        if dev & ext:
            raise ValueError(f"Target-task external structure leakage: {task}")
    if reference["measurement_encoding_mode"] != "exact_only":
        raise ValueError("The reference model is not exact_only")
    if reference["ensemble_size"] != 3 or reference["best_epoch"] != 32:
        raise ValueError("Reference final ensemble configuration changed")


def check_release() -> tuple[pd.DataFrame, dict]:
    verify_manifest()
    data = pd.read_csv(DATA_PATH, low_memory=False)
    reference = torch.load(REFERENCE_PATH, map_location="cpu", weights_only=True)
    params = json.loads(PARAM_PATH.read_text(encoding="utf-8"))
    if params != reference["best_params"]:
        raise ValueError("Fixed hyperparameters differ from the reference model")
    verify_data(data, reference)
    frozen = torch.load(FROZEN_PATH, map_location="cpu", weights_only=True)
    if frozen.get("format") != "ctml_frozen_lc50_expert_v1":
        raise ValueError("Not a frozen structural-expert checkpoint")
    mapping = {
        "node_input": "lc50_expert_node_input.",
        "graph_encoder": "lc50_expert_graph_encoder.",
        "pool": "lc50_expert_pool.",
    }
    for module_name, prefix in mapping.items():
        for key, value in frozen["module_states"][module_name].items():
            full_name = prefix + key
            for i, state in enumerate(reference["state_dicts"]):
                if not torch.equal(value, state[full_name]):
                    raise ValueError(f"LC50 frozen expert mismatch, ensemble={i}, key={full_name}")
    if int(frozen["species_vocab_size"]) != int(
        reference["state_dicts"][0]["lc50_species_embedding.weight"].shape[0]
    ):
        raise ValueError("Frozen expert species vocabulary mismatch")
    return data, params


def print_external_effect_r2(predictions: pd.DataFrame) -> None:
    from scripts.external_task_metrics import summarize_by_effect

    metrics = summarize_by_effect(predictions)
    print(metrics[["effect", "R2"]].to_string(
        index=False, formatters={"R2": lambda value: f"{value:.6f}"}
    ))


def run_child(arguments: list[str], log_path: Path) -> None:
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), *arguments],
            cwd=str(ROOT),
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode:
        details = log_path.read_text(encoding="utf-8", errors="replace")
        raise RuntimeError(f"CTML process failed (exit code {result.returncode}):\n{details[-12000:]}")


def train_worker(output: Path, device: str) -> None:
    frame, fixed_params = check_release()
    run_multitask_training(
        frame,
        output_dir=output / "model",
        fixed_params=fixed_params,
        seed=FIXED_CONFIG["seed"],
        cv_folds=FIXED_CONFIG["cv_folds"],
        cv_repeats=FIXED_CONFIG["cv_repeats"],
        max_epochs=FIXED_CONFIG["max_epochs"],
        patience=FIXED_CONFIG["patience"],
        final_ensemble_seeds=FIXED_CONFIG["final_ensemble_seeds"],
        device_name=device,
        num_workers=FIXED_CONFIG["num_workers"],
        use_uncertainty_weighting=True,
        use_pcgrad=False,
        uncertainty_lr_multiplier=FIXED_CONFIG["uncertainty_lr_multiplier"],
        uncertainty_warmup_epochs=FIXED_CONFIG["uncertainty_warmup_epochs"],
        progress_every=5,
        use_cross_task_evidence=True,
        use_record_level_evidence=True,
        use_task_atom_attention=True,
        use_auxiliary_loss=True,
        use_oof_calibration=True,
        calibration_strength=FIXED_CONFIG["calibration_strength"],
        use_lc50_auxiliary=True,
        lc50_frozen_expert_checkpoint=str(FROZEN_PATH),
        lc50_joint_steps=0,
        use_fingerprint_expert=True,
        use_evidence_attention=True,
        use_record_quality_gate=True,
        use_task_residual=True,
        measurement_encoding_mode=FIXED_CONFIG["measurement_encoding_mode"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--train", action="store_true")
    mode.add_argument("--evaluate", action="store_true")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--_train_worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--_temp_output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args._train_worker:
        if not args.train or args._temp_output is None:
            parser.error("Invalid worker invocation")
        train_worker(args._temp_output, args.device)
        return

    if args.check_only:
        check_release()
        return

    check_release()
    with tempfile.TemporaryDirectory(prefix="ctml_run_") as temporary:
        temp = Path(temporary)
        log = temp / "process.log"
        if args.train:
            run_child(
                ["--train", "--device", args.device, "--_train_worker", "--_temp_output", str(temp)],
                log,
            )
            predictions = temp / "model" / "predictions_external.csv"
        else:
            predictions = temp / "external_predictions.csv"
            with log.open("w", encoding="utf-8") as log_file:
                result = subprocess.run(
                    [
                        sys.executable,
                        str(ROOT / "scripts" / "evaluate_ctml.py"),
                        "--device", args.device,
                        "--out", str(predictions),
                    ],
                    cwd=str(ROOT),
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            if result.returncode:
                details = log.read_text(encoding="utf-8", errors="replace")
                raise RuntimeError(f"CTML evaluation failed:\n{details[-12000:]}")
        if not predictions.is_file():
            raise FileNotFoundError("External-validation predictions were not generated")
        print_external_effect_r2(pd.read_csv(predictions))


if __name__ == "__main__":
    main()
