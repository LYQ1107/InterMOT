#!/usr/bin/env python3
"""Strict 6/1/1 R3R2 metric adapters and honest historic checkpoint reuse."""
from __future__ import annotations
import argparse
from scripts.n72r20r4_common import *
from scripts import n72r20r3r2_representation as r3
from sam3_intermot.association.identity_authority import AdapterEnsemble


def fold_split(heldout: str) -> tuple[list[str], str]:
    validation = SEQUENCES[(SEQUENCES.index(heldout)+1) % len(SEQUENCES)]
    return [s for s in SEQUENCES if s not in {heldout, validation}], validation


def historical_ensemble(heldout: str) -> AdapterEnsemble:
    records = [r for r in read_json(OUT / "adapter/CHECKPOINT_AUDIT.json")["records"] if r["heldout_sequence"] == heldout]
    ensemble = AdapterEnsemble([Path(r["path"]) for r in records], expected_shas=[r["sha256"] for r in records], forbidden_sequences=[heldout])
    for actual, record in zip(ensemble.manifest, records):
        actual["actual_training_sequences"] = record["actual_training_sequences_from_source"]
        actual["metadata_discrepancy_corrected_by_source_audit"] = True
    return ensemble


def train(heldouts=SEQUENCES) -> dict:
    torch.set_num_threads(1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = None
    results = {}
    for heldout in heldouts:
        fit, validation = fold_split(heldout)
        records = []
        for seed in SEEDS:
            path = ASSETS / "models" / f"adapter6__{heldout}__seed{seed}.pt"
            done = path.with_suffix(".json")
            if done.exists():
                record = read_json(done)
                if sha256(path) != record["sha256"] or record["actual_training_sequences"] != fit:
                    raise ValueError("invalid adapter training resume")
                records.append(record)
                continue
            check_storage(reserve_gib=0.04)
            if dataset is None:
                dataset = r3.load_episode_dataset()
            fit_episodes = [e for e in dataset.episodes if e["sequence"] in fit]
            inner_episodes = [e for e in dataset.episodes if e["sequence"] == validation]
            model, training = r3._fit_inner(dataset, fit_episodes, inner_episodes, seed, device)
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"stage": STAGE, "source_architecture_stage": "N72R20R3R2", "architecture": "CrossSceneIdentityAdapter", "feature_dim": 512, "bottleneck_dim": 128, "temperature": 0.07, "heldout_sequence": heldout, "parameter_fit_sequences": fit, "actual_training_sequences": fit, "internal_validation_sequence": validation, "seed": seed, "trainable_parameters": model.trainable_parameters, "OSNet_frozen": True, "GRU_frozen": True, "runtime_future_gt_used": False, "state_dict": {k:v.detach().cpu() for k,v in model.state_dict().items()}}, path)
            record = {"path": path, "sha256": sha256(path), "seed": seed, "heldout_sequence": heldout, "actual_training_sequences": fit, "internal_validation_sequence": validation, "fit_episode_count": len(fit_episodes), "inner_episode_count": len(inner_episodes), "outer_heldout_absent": True, "parameters": model.trainable_parameters, "training": training, "code_sha256": code_manifest()}
            write_json(done, record)
            records.append(plain(record))
            print(json.dumps({"heldout": heldout, "seed": seed, "epoch": training["best_epoch"], "status": "ADAPTER6_TRAINED"}), flush=True)
            del model
        results[heldout] = records
        write_json(OUT / "adapter/STRICT_FOLD_CHECKPOINTS.json", {"stage": STAGE, "completed_folds": results, "seeds": SEEDS, "strict_6_1_1": True, "backbone_training": False, "reason": "historic seven-sequence final refit contains current inner sequence"})
    return results


def strict_ensemble(heldout: str, seeds=SEEDS) -> AdapterEnsemble:
    fit, validation = fold_split(heldout)
    paths = [ASSETS / "models" / f"adapter6__{heldout}__seed{seed}.pt" for seed in seeds]
    records = [read_json(p.with_suffix(".json")) for p in paths]
    return AdapterEnsemble(paths, expected_shas=[r["sha256"] for r in records], forbidden_sequences=[heldout, validation])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--heldouts", nargs="+", default=list(SEQUENCES))
    args = parser.parse_args()
    print(json.dumps({"completed_folds": list(train(args.heldouts))}))
