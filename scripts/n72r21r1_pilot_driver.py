"""Bounded CPU scheduler; task-owned subprocesses, no foreign process changes."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess
import os
import sys
from scripts.n72r21r1_common import ROOT, OUT, ASSETS, read_json, write_json, sha256, storage


def worker(sequence, seed):
    command = [sys.executable, str(ROOT / 'scripts/n72r21r1_fixed_pilot.py'), '--sequence', sequence, '--seed', str(seed)]
    environment = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', PYTHONPATH=str(ROOT))
    log_path = ASSETS / 'fixed_pilot/logs' / (sequence + '__seed' + str(seed) + '.log')
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists(): raise FileExistsError('retain scheduler log; use case script for evidence-verified resume')
    with log_path.open('x') as log:
        result = subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, check=False)
    return {'sequence': sequence, 'seed': seed, 'command': command, 'returncode': result.returncode, 'log_path': str(log_path), 'log_sha256': sha256(log_path)}


if __name__ == '__main__':
    storage(512 << 20)
    protocol = read_json(OUT / 'protocol/FIXED_GATE_PILOT.json'); results = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        jobs = [executor.submit(worker, sequence, seed) for seed in protocol['historical_weight_reproduction_seeds'] for sequence in protocol['sequences']]
        for job in as_completed(jobs):
            result = job.result(); results.append(result)
            print({'fixed_pilot_worker_terminal': result['sequence'], 'seed': result['seed'], 'returncode': result['returncode']}, flush=True)
    write_json('pilot/SCHEDULER_RESULT.json', {'workers': results, 'actual_max_concurrent_workers': 4, 'status': 'ALL_WORKERS_TERMINAL_NOT_SCIENTIFIC_PASS'})
