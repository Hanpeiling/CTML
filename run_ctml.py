from __future__ import annotations

import argparse
import contextlib
import io
import tempfile
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
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
        p = ROOT / relative
        if not p.is_file():
            raise FileNotFoundError(f"Required release asset missing: {relative}")
        actual = sha256(p)
        if actual != expected:
            raise ValueError(f"Release asset checksum mismatch: {relative}")
    print(f"Verified {len(manifest)} SHA256 checksums")


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
    print("Verified: 5005 curated records, 67 species, 8 effects, 108 task-specific indicators")
    print("Verified: fixed train/test/external labels and task-specific external chemical holdouts")


def check_release() -> tuple[pd.DataFrame, dict]:
    verify_manifest()
    data = pd.read_csv(DATA_PATH, low_memory=False)
    ref = torch.load(REFERENCE_PATH, map_location="cpu", weights_only=True)
    params = json.loads(PARAM_PATH.read_text(encoding="utf-8"))
    if params != ref["best_params"]:
        raise ValueError("Fixed hyperparameters differ from the reference model")
    verify_data(data, ref)
    frozen = torch.load(FROZEN_PATH, map_location="cpu", weights_only=True)
    if frozen.get("format") != "ctml_frozen_lc50_expert_v1":
        raise ValueError("Not a frozen structural-expert checkpoint")
    mapping = {
        "node_input": "lc50_expert_node_input.",
        "graph_encoder": "lc50_expert_graph_encoder.",
        "pool": "lc50_expert_pool.",
    }
    for module_name, prefix in mapping.items():
        for k, v in frozen["module_states"][module_name].items():
            full_name = prefix + k
            for i, state in enumerate(ref["state_dicts"]):
                if not torch.equal(v, state[full_name]):
                    raise ValueError(f"LC50 frozen expert mismatch, ensemble={i}, key={full_name}")
    assert int(frozen["species_vocab_size"]) == int(
        ref["state_dicts"][0]["lc50_species_embedding.weight"].shape[0]
    )
    print("Verified: frozen LC50 expert exactly matches all three reference ensemble members")
    print("Verified: fixed CTML model hyperparameters and configuration")
    print("No training has been performed by --check-only")
    return data, params


def print_external_effect_r2(predictions: pd.DataFrame) -> None:
    """Print only the eight task-specific external-validation R2 values."""
    from scripts.external_task_metrics import summarize_by_effect

    metrics = summarize_by_effect(predictions)
    
    print(metrics[["effect", "R2"]].to_string(
        index=False, formatters={"R2": lambda v: f"{v:.6f}"}
    ))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    task = parser.add_mutually_exclusive_group(required=True)
    task.add_argument("--check-only", action="store_true", help="Validate all release assets without training")
    task.add_argument("--train", action="store_true", help="Run 2 x 10-fold CV and train the final ensemble")
    task.add_argument("--evaluate", action="store_true", help="Evaluate the released CTML ensemble on all external records")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output", default="outputs_CTML", help="Output directory")
   
    parser.add_argument("--_train_worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.train and not args._train_worker:
       
        output = Path(args.output).expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            str(ROOT / "run_ctml.py"),
            "--train",
            "--device", args.device,
            "--output", str(output),
            "--_train_worker",
        ]
        console_log = output / "training_console.log"
        with console_log.open("w", encoding="utf-8") as log:
            result = subprocess.run(
                cmd,
                cwd=str(ROOT),
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if result.returncode != 0:
            raise RuntimeError(
                f"CTML training failed (exit code {result.returncode}). "
                f"See {console_log} and {output / 'training.log'} for details."
            )
        predictions_file = output / "model" / "predictions_external.csv"
        if not predictions_file.is_file():
            raise FileNotFoundError(
                f"Training completed, but external predictions are missing: {predictions_file}"
            )
        print_external_effect_r2(pd.read_csv(predictions_file))
        return

    
    if args.evaluate or args.train:
        with contextlib.redirect_stdout(io.StringIO()):
            frame, fixed_params = check_release()
    else:
        frame, fixed_params = check_release()
    if args.check_only:
        return
    output = Path(args.output).expanduser().resolve()
    if args.evaluate:
     
        with tempfile.TemporaryDirectory(prefix="ctml_external_eval_") as temp_dir:
            predictions_path = Path(temp_dir) / "external_predictions.csv"
            cmd = [
                sys.executable, str(ROOT / "scripts" / "evaluate_ctml.py"),
                "--device", args.device,
                "--out", str(predictions_path),
            ]
            result = subprocess.run(
                cmd, cwd=str(ROOT), capture_output=True, text=True,
                errors="replace", check=False,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    "CTML evaluation failed. Original error output:\n"
                    + (result.stderr or result.stdout or "Unknown error")
                )
            print_external_effect_r2(pd.read_csv(predictions_path))
        return
    output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "released_config": FIXED_CONFIG,
        "input_data_sha256": sha256(DATA_PATH),
        "frozen_lc50_sha256": sha256(FROZEN_PATH),
        "fixed_hyperparameters_sha256": sha256(PARAM_PATH),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_requested": args.device,
    }
    import importlib.metadata as im
    for pkg in ["numpy", "pandas", "scipy", "scikit-learn", "optuna", "rdkit", "joblib"]:
        try: metadata[pkg] = im.version(pkg)
        except im.PackageNotFoundError: metadata[pkg] = "not installed"
    (output / "environment_and_inputs.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    
    with (output / "training.log").open("w", encoding="utf-8") as log, contextlib.redirect_stdout(log):
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
            device_name=args.device,
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
    predictions_file = output / "model" / "predictions_external.csv"
    if not predictions_file.is_file():
        raise FileNotFoundError(
            "Training completed without external predictions: "
            + str(predictions_file)
        )
    
    print_external_effect_r2(pd.read_csv(predictions_file))


if __name__ == "__main__":
    main()
