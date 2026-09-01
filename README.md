# InterMOT

InterMOT is an interactive multi-object tracking research system built around
the official Meta SAM 3.1 Object Multiplex video model.

The code provides:

- a stable backend abstraction for SAM3 video inference;
- multi-object tracking with explicit track lifecycle management;
- separate SAM object, MOT track, and identity-lineage identifiers;
- auditable Add, Correct, Reassign, and Delete interaction transactions;
- MOTChallenge/TrackEval integration;
- a GT-driven interaction simulator for controlled, non-production experiments.

## Repository scope

This public repository contains the source package, tests, experiment scripts,
environment templates, and a path-portable configuration template. Local
checkpoints, datasets, generated outputs, caches, private audit logs, and
machine-specific manifests are intentionally excluded.

The project has no project-level open-source license declared yet. The SAM3
dependency is referenced as an official submodule and remains subject to its
own license and usage terms.

## Requirements

- Python 3.12
- PyTorch and CUDA versions compatible with the installed official SAM3 release
- Access to the official SAM3 checkpoint, if running real inference
- A local DanceTrack dataset for dataset-backed experiments

The checkpoint is not included. Request access through the official
[SAM3 repository](https://github.com/facebookresearch/sam3) and follow its
checkpoint terms.

## Setup

Clone the repository with its pinned official SAM3 dependency:

\`\`\`bash
git clone --recurse-submodules https://github.com/LYQ1107/InterMOT.git
cd InterMOT
conda env create -f environment/environment.yml
conda activate sam3_intermot
pip install -e third_party/sam3
\`\`\`

Alternatively, install the Python dependencies listed in
environment/requirements-lock.txt inside an existing environment.

Set the checkpoint and dataset paths in configs/default.yaml before a real
run. The committed values are placeholders and do not identify a local
machine.

## Tests and mock pipeline

The deterministic mock backend can be exercised without a SAM3 checkpoint:

\`\`\`bash
PYTHONPATH=. python -m pytest -q
PYTHONPATH=. python scripts/run_mock_pipeline.py
\`\`\`

The mock path is for software validation only; its output is not a real SAM3
tracking result.

## Reproducibility and data use

Experiment scripts may require local dataset/checkpoint paths and are kept
separate from generated results. Runtime code must not use future ground truth
for online tracking decisions; GT is reserved for offline labels and
post-hoc evaluation in controlled experiments.

## Acknowledgements

This project uses the official Meta SAM3 implementation as an external
dependency. Please consult the submodule's license and upstream documentation
before redistribution or commercial use.
