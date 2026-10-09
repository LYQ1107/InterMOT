"""Two bounded FIT corpus workers, alongside at most two existing diagnostics."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import sys
import subprocess
from scripts.n72r21r1_common import ROOT, OUT, ASSETS, read_json, write_json, sha256, storage


def worker(sequence):
    command = [sys.executable, str(ROOT / 'scripts/n72r21r1_collect_joint.py'), '--sequence', sequence]
    path = ASSETS / 'corpus/logs' / (sequence + '.log'); path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists(): raise FileExistsError('keep prior worker log and verify live process before case resume')
    environment = dict(os.environ, PYTHONPATH=str(ROOT), OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    with path.open('x') as log:
        result = subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, check=False)
    return {'sequence': sequence, 'command': command, 'returncode': result.returncode, 'log_path': str(path), 'log_sha256': sha256(path)}


if __name__ == '__main__':
    storage(1 << 30)
    protocol = read_json(OUT / 'protocol/JOINT_STATE_CORPUS.json'); results = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        jobs = [executor.submit(worker, s) for s in protocol['identity_model_FIT_and_new_controller_diagnostic_FIT']]
        for job in as_completed(jobs):
            value = job.result(); results.append(value)
            print({'FIT_causal_corpus_worker_terminal': value['sequence'], 'returncode': value['returncode']}, flush=True)
    write_json('corpus/FIT_SCHEDULER_RESULT.json', {'workers': results, 'concurrent_FIT_workers_max': 2,
        'not_model_training': True, 'not_heldout_confirmation': True, 'status': 'WORKERS_TERMINAL_CHECK_ACTUAL_SEALS'})
