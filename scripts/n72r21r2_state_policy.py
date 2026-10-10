"""Full own-video MOT for the registered9 matched state-source fits.

Reuses the frozen MAIN replay/evaluation implementation in a process-local
adapter. It does not edit that implementation, tracker, identity memory or
training. Controlled treatment supervision is NOT learned on-policy closure.
"""
import argparse
from contextlib import contextmanager
from copy import deepcopy
from scripts import n72r21r2_main_policy as engine
from scripts.n72r21r2_train_joint_state import PROTOCOL as FIT_PROTOCOL, MODES, OBJECTIVE
from scripts.n72r21r2_common import (ROOT, OUT, ASSETS, GOAL, read_json, write_json, sha256,
    preregistration, development_sequence, storage)
from sam3_intermot.evaluation.learned_policy_evidence import choose_inner_point
from sam3_intermot.one_click.runtime_file_guard import runtime_file_guard

PROTOCOL = OUT / "on_policy/STATE_POLICY_PROTOCOL_V1.json"
PREFIX = "mot/state_policy_v1"
BASE = ASSETS / "state_policy_v1"
CODE = tuple(dict.fromkeys((*engine.CODE, "scripts/n72r21r2_state_policy.py",
    "scripts/n72r21r2_state_policy_driver.py", "scripts/n72r21r2_train_joint_state.py",
    "scripts/n72r21r2_joint_state_fit_driver.py")))


def freeze():
    source, replay = read_json(FIT_PROTOCOL), read_json(engine.PROTOCOL)
    assert source["modes"] == list(MODES) and source["family"] == "SMALL_MLP" and source["objective"] == OBJECTIVE
    assert source["seeds"] == replay["seeds"]
    assert not list((OUT / "on_policy/state_source_fits_v1").glob("*STATE__seed*.json")), "Freeze before state-source fits/outcomes"
    storage(2 << 30)
    write_json("on_policy/STATE_POLICY_PROTOCOL_V1.json", {"stage": "N72R21R2", "goal": GOAL,
        "goal_file": "outputs/N72R21R2/FINAL_GOAL.json", "frozen": True, "modes": source["modes"],
        "family": source["family"], "objective": source["objective"], "seeds": source["seeds"],
        "split": {k: source["split"][k] for k in ("fit", "inner")}, "points": replay["points"],
        "authority_support_policy": replay["authority_support_policy"],
        "source_sha256": {name: sha256(ROOT / name) for name in CODE},
        "state_fit_protocol_sha256": sha256(FIT_PROTOCOL),
        "inherited_MAIN_replay_protocol_sha256": sha256(engine.PROTOCOL),
        "replay_and_selection": "Same raw anchor/current candidate axis, frozen P0 memory, full causal global assignment, strict NONE/hard negatives, own-prefix cloned one-shot all-action H100 audits, actual pinned nine-metric whole video and same shared all3-seed INNER points / selected FIT16 as MAIN. No new optimization, hyperparameters or best seed.",
        "training_source": "Controlled BASELINE_STATE/TREATMENT_STATE/MIXED_STATE matched source contrast. It is not actual model-generated on-policy or staged-policy learning.",
        "zero_or_failed_fits": "Actual no-benefit/harm supervision failure is retained, not a weight; do not synthesize/refit missing seeds or select a successful subset.",
        "qualified_model_generated_rounds_remain_separate": True,
        "independent_G1_root_proof_required_separately": True,
        "confirmation_VAL_TEST_SOT_authorized": False, "CPU_workers": 1, "OMP_threads": 1,
        "estimated_extra_assets_GiB_planning": 2., "minimum_free_GiB": 60,
        "resource": "1 CPU worker only after all24 controlled source receipts. Estimated2GiB includes9/42 MAIN trace/trajectory/evaluation scale plus onset room; measured runtime growth/60GiB floor enforced, no deletion of unique sources."})


def inputs(sequence, mode, objective, seed, point):
    development_sequence(sequence)
    p = read_json(PROTOCOL)
    assert p["goal"] == GOAL and p["source_sha256"] == {name: sha256(ROOT / name) for name in CODE}
    assert p["state_fit_protocol_sha256"] == sha256(FIT_PROTOCOL)
    assert p["inherited_MAIN_replay_protocol_sha256"] == sha256(OUT / "mot/MAIN_POLICY_PROTOCOL_V1.json")
    if mode not in p["modes"] or objective != p["objective"] or seed not in p["seeds"] or point not in p["points"]:
        raise ValueError("Only fixed9 state-source models / inherited2 points may replay")
    path = OUT / "on_policy/state_source_fits_v1" / (mode + "__seed" + str(seed) + ".json")
    fit = read_json(path)
    assert fit["status"] == "COMPLETE_ACTUAL_STATE_SOURCE_CONTRAST_UNCALIBRATED_NOT_DEPLOYABLE"
    assert fit["mode"] == mode and fit["family"] == "SMALL_MLP" and fit["objective"] == OBJECTIVE
    assert not fit["actual_model_generated_on_policy"] and not fit["association_authority"]
    assert fit["nonzero_gradient_steps"] > 0 and fit["changed_weight_elements"] > 0
    assert sha256(fit["checkpoint_path"]) == fit["checkpoint_sha256"]
    assert fit["protocol_sha256"] == sha256(FIT_PROTOCOL)
    assert fit["source_sha256"] == read_json(FIT_PROTOCOL)["source_sha256"]
    assert fit["source_sha256"] == {name: sha256(ROOT / name) for name in fit["source_sha256"]}
    assert set(fit["manifest"]["actual_FIT_videos"]).issubset(p["split"]["fit"])
    assert set(fit["manifest"]["actual_INNER_videos"]).issubset(p["split"]["inner"])
    if sequence in p["split"]["fit"]:
        selected = read_json(OUT / PREFIX / "selections" / (mode + "__" + OBJECTIVE + ".json"))
        assert selected["selected_point"] == point and selected["protocol_sha256"] == sha256(PROTOCOL)
    uid = "STATE__" + mode + "__" + OBJECTIVE + "__seed" + str(seed) + "__" + point
    return p, uid, fit, path


@contextmanager
def scoped_engine():
    """Every patched process-local symbol is restored even on failure."""
    names = ("PROTOCOL", "PREFIX", "BASE", "CODE", "inputs", "write_json")
    previous = {name: getattr(engine, name) for name in names}
    def state_write(relative, value, **kwargs):
        return write_json(relative, {**value, "actual_architecture_family": "SMALL_MLP",
            "controlled_training_state_source_contrast": True, "actual_model_generated_on_policy_training": False,
            "M7_on_policy_or_staged_closure_NOT_inferred_from_THIS_contrast": True}, **kwargs)
    try:
        engine.PROTOCOL, engine.PREFIX, engine.BASE, engine.CODE = PROTOCOL, PREFIX, BASE, CODE
        engine.inputs, engine.write_json = inputs, state_write
        yield
    finally:
        for name, value in previous.items():
            setattr(engine, name, value)


def execute(action, sequence, mode, seed, point):
    if action not in ("runtime", "onsets", "evaluate"):
        raise ValueError("Registered full-policy operation required")
    with scoped_engine():
        worker = {"runtime": engine.runtime, "onsets": engine.onset_runtime, "evaluate": engine.evaluate}[action]
        if action == "evaluate":
            worker(sequence, mode, OBJECTIVE, seed, point)
        else:
            with runtime_file_guard():
                worker(sequence, mode, OBJECTIVE, seed, point)


def select(mode):
    p = read_json(PROTOCOL)
    assert mode in p["modes"] and p["source_sha256"] == {name: sha256(ROOT / name) for name in CODE}
    cells, refs = {}, []
    for point in p["points"]:
        cells[point] = []
        for seed in p["seeds"]:
            for sequence in p["split"]["inner"]:
                _, uid, _, _ = inputs(sequence, mode, OBJECTIVE, seed, point)
                path = OUT / PREFIX / "results" / uid / (sequence + ".json")
                cell = read_json(path)
                assert cell["protocol_sha256"] == sha256(PROTOCOL)
                assert cell["actual_architecture_family"] == "SMALL_MLP" and cell["controlled_training_state_source_contrast"]
                cells[point].append(cell)
                refs.append({"path": str(path), "sha256": sha256(path)})
    chosen = choose_inner_point(cells, p["seeds"], p["split"]["inner"])
    write_json(PREFIX + "/selections/" + mode + "__" + OBJECTIVE + ".json", {"stage": "N72R21R2", "goal": GOAL,
        "protocol_sha256": sha256(PROTOCOL), "mode": mode, "family": "SMALL_MLP", "objective": OBJECTIVE,
        "actual_INNER_result_refs": refs, "actual_model_generated_on_policy": False, **chosen})
    print({"actual_state_source_INNER_selection": mode, "point": chosen["selected_point"], "status": chosen["status"]}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("freeze", "runtime", "onsets", "evaluate", "select"))
    parser.add_argument("--sequence")
    parser.add_argument("--mode", choices=MODES)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--point")
    args = parser.parse_args()
    if args.action == "freeze":
        freeze()
    elif args.action == "select":
        select(args.mode)
    else:
        execute(args.action, args.sequence, args.mode, args.seed, args.point)
