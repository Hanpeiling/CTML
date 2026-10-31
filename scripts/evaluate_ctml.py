from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
import torch

from config import TASK_TO_ID, GROUP_COL
from models import MultiTaskAttentionRegressor
from preprocessing import ContextPreprocessor
from scripts.external_task_metrics import save_and_print
from multitask_training import (
    TaskTargetScaler, ContextAwareRecordEvidenceStore,
    _make_loader, predict_loader, apply_task_calibration,
    TRANSFER_FEATURE_DIM, RECORD_TRANSFER_FEATURE_DIM,
)


def load_checkpoint(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def make_model(ck, member_state):
    params = ck["best_params"]
    vocab = int(member_state["lc50_species_embedding.weight"].shape[0])
    model = MultiTaskAttentionRegressor(
        context_dims=ck["context_dims"],
        hidden_dim=int(params["hidden_dim"]),
        n_layers=int(params["n_layers"]),
        num_heads=int(params.get("num_heads", 1)),
        dropout=float(params["dropout"]),
        n_tasks=len(TASK_TO_ID),
        transfer_feature_dim=TRANSFER_FEATURE_DIM,
        record_transfer_feature_dim=RECORD_TRANSFER_FEATURE_DIM,
        lc50_species_vocab_size=vocab,
        use_lc50_auxiliary=False,
        use_lc50_expert=True,
        record_attention_temperature=float(params.get("record_attention_temperature", 0.40)),
        use_cross_task_evidence=bool(ck["use_cross_task_evidence"]),
        use_record_level_evidence=bool(ck["use_record_level_evidence"]),
        use_task_atom_attention=bool(ck["use_task_atom_attention"]),
        use_fingerprint_expert=bool(ck["use_fingerprint_expert"]),
        use_evidence_attention=bool(ck["use_evidence_attention"]),
        use_record_quality_gate=bool(ck["use_record_quality_gate"]),
        use_task_residual=bool(ck["use_task_residual"]),
    )
    model.load_state_dict(member_state, strict=True)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional first N external records; omit for all 328")
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    parser.add_argument("--out", type=Path, default=ROOT/"outputs_CTML"/"external_predictions.csv")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    frame = pd.read_csv(ROOT/"data"/"curated_data_with_splits.csv", low_memory=False)
    development = frame[frame["split"]!="external"].copy()
    external = frame[frame["split"]=="external"].copy()
    if args.limit:
        external = external.iloc[:args.limit].copy()
    ck = load_checkpoint(ROOT/"model"/"multitask_model.pt")
    if ck["measurement_encoding_mode"] != "exact_only":
        raise ValueError("Reference checkpoint is not exact_only")
    
    prep = ContextPreprocessor(measurement_encoding_mode="exact_only").fit(development)
    if prep.output_dims != ck["context_dims"]:
        raise ValueError(
            f"Preprocessor dimensions differ: rebuilt={prep.output_dims}, "
            f"reference={ck['context_dims']}. Reproduce with scikit-learn 1.3.2."
        )
    ev_store = joblib.load(ROOT/"model"/"cross_task_evidence_store.joblib")
    s=ck["task_target_scalers"]
    scaler = TaskTargetScaler(
        global_mean=float(s["global_mean"]),
        global_std=float(s["global_std"]),
        task_stats={k: (float(v["mean"]),float(v["std"])) for k,v in s["task_stats"].items()},
    )
    record_store = ContextAwareRecordEvidenceStore.fit(
        development, prep, scaler,
        per_source_task=int(ck["best_params"].get("record_evidence_per_source",4)),
        n_tasks=len(TASK_TO_ID),
    )
    device=torch.device("cuda" if args.device=="auto" and torch.cuda.is_available() else
                        "cpu" if args.device=="auto" else args.device)
    print(f"Preparing external prediction loader for {len(external)} records on {device}...",flush=True)
    loader = _make_loader(
        external, prep, scaler, int(ck["best_params"]["batch_size"]), False,
        num_workers=0,
        evidence_store=ev_store,
        record_evidence_store=record_store,
        use_cross_task_evidence=True,
        use_record_level_evidence=True,
        record_evidence_per_source=int(ck["best_params"]["record_evidence_per_source"]),
        task_sampling_power=float(ck["best_params"]["task_sampling_power"]),
        seed=42,
    )
    preds=[]
    with torch.no_grad():
        for i,sd in enumerate(ck["state_dicts"]):
            model=make_model(ck,sd).to(device)
            p=predict_loader(model,loader,device,scaler,return_attention=False)
            preds.append(p.sort_values("row_index").reset_index(drop=True))
            print(f"Reference ensemble member {i+1}/{len(ck['state_dicts'])} completed",flush=True)
    out=preds[0][["row_index","task","true_log10"]].copy()
    arr=np.stack([p["pred_log10"].to_numpy(dtype=float) for p in preds],axis=1)
    out["pred_log10"]=arr.mean(axis=1)
    out["pred_log10_sd"]=arr.std(axis=1,ddof=0)
    out=apply_task_calibration(out,pd.DataFrame(ck["oof_calibration"]))
   
    y=out["true_log10"].to_numpy(dtype=float)
    pred=out["pred_log10"].to_numpy(dtype=float)
    if not np.isfinite(y).all() or not np.isfinite(pred).all():
        raise ValueError("Non-finite observed or predicted toxicity values")
    r2 = 1.0 - float(np.sum((y - pred) ** 2) / np.sum((y - np.mean(y)) ** 2))
    rmse = float(np.sqrt(np.mean((y - pred) ** 2)))
    mae = float(np.mean(np.abs(y - pred)))
    args.out.parent.mkdir(parents=True,exist_ok=True)
    out.to_csv(args.out,index=False)
    save_and_print(out[["task","true_log10","pred_log10"]],
                   args.out.with_name("external_metrics_by_effect.csv"))
    print(f"Evaluated {len(out)} external records with the released CTML model")
    print(f"External validation: n={len(out)}, R2={r2:.6f}, RMSE={rmse:.6f}, MAE={mae:.6f}")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
