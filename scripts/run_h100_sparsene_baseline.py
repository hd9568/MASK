#!/usr/bin/env python3
"""Generate and benchmark the non-agent SR-BCRS path in an isolated run."""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'python/test')]
from run_builtin_backend_e2e import generate_candidate


def run_logged(command: list[str], log: Path, *, timeout: int | None = None) -> int:
    with log.open('w', encoding='utf-8') as out:
        result = subprocess.run(command, cwd=ROOT, stdout=out, stderr=subprocess.STDOUT,
                                timeout=timeout, check=False)
    return result.returncode


def choose_calibrated(candidates: list[dict], *, minimum_gain: float = 0.015) -> dict | None:
    """Select a validated median only when it clearly improves on baseline."""
    baseline = next((x for x in candidates if x['name'] == 'baseline' and x['valid']), None)
    if baseline is None:
        return None
    valid = [x for x in candidates if x['valid']]
    best = min(valid, key=lambda x: x['median_ms'])
    return best if best['median_ms'] < baseline['median_ms'] * (1.0 - minimum_gain) else baseline


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True, help='New, empty output directory')
    p.add_argument('--matrix-dir', type=Path, required=True)
    p.add_argument('--cutlass-root', type=Path, required=True)
    p.add_argument('--cuda-arch', choices=('80', '90'), default='90')
    p.add_argument('--host-compiler', default='/usr/bin/g++')
    p.add_argument('--n', type=int, default=512)
    p.add_argument('--warmup', type=int, default=3)
    p.add_argument('--repeat', type=int, default=32)
    p.add_argument('--trials', type=int, default=3)
    p.add_argument('--timeout', type=int, default=600)
    p.add_argument('--datasets', nargs='+', help='Optional matrix names without .mtx')
    p.add_argument('--policy', choices=('auto', 'row', 'strict_lb', 'multi_bind'), default='auto')
    p.add_argument('--task-group-tiles', type=int, default=64)
    p.add_argument('--autotune-policy', action='store_true', help='Calibrate policies on this GPU before separate evaluation')
    p.add_argument('--calibration-trials', type=int, default=2)
    a = p.parse_args()
    if min(a.n, a.repeat, a.trials, a.timeout, a.task_group_tiles, a.calibration_trials) < 1 or a.warmup < 0:
        p.error('positive n/repeat/trials/timeout/task-group-tiles and nonnegative warmup required')
    if a.autotune_policy and a.policy != 'auto':
        p.error('autotune-policy requires the default auto policy')
    if a.n % 64:
        p.error('n must be divisible by the generated tile width 64')
    if not (a.cutlass_root / 'include/cute/tensor.hpp').is_file():
        p.error('cutlass root must contain include/cute/tensor.hpp')
    names = a.datasets or sorted(x.stem for x in a.matrix_dir.glob('*.mtx'))
    if not names or len(names) != len(set(names)):
        p.error('matrix selection must be nonempty and contain no duplicates')
    for name in names:
        if Path(name).name != name or not (a.matrix_dir / (name + '.mtx')).is_file():
            p.error('invalid or missing dataset: ' + name)
    run = a.run_dir.resolve()
    run.mkdir(parents=True, exist_ok=False)
    work = run / 'candidate'
    work.mkdir()
    manifest = generate_candidate(work, tile_n=64, max_tiles=32)
    cfg = ['cmake', '-S', str(ROOT / 'examples/src_fp32/acc/testbed'), '-B', str(work / 'build'),
           '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_CUDA_HOST_COMPILER=' + a.host_compiler,
           '-DSPARSENE_CUSTOM_CUDA_ARCHITECTURES=' + a.cuda_arch,
           '-DSPARSENE_CUTLASS_ROOT=' + str(a.cutlass_root.resolve()),
           '-DSPARSENE_GENERATED_OPERATOR=' + manifest['operator_path'],
           '-DSPARSENE_LOGICAL_BLK_M=16', '-DSPARSENE_PHYSICAL_BLK_M=16',
           '-DSPARSENE_BLK_K=8', '-DSPARSENE_TILE_N=64', '-DSPARSENE_ROW_REORDER=0',
           '-DSPARSENE_PACKING_MODE=2', '-DSPARSENE_MAX_TILES_PER_SEGMENT=32']
    for cmd, log in ((cfg, 'configure.log'),
                     (['cmake', '--build', str(work / 'build'), '--target', 'custom-format-e2e', '-j1'], 'build.log')):
        rc = run_logged(cmd, run / log)
        if rc:
            print(f'{log} failed ({rc}); see {run / log}', file=sys.stderr)
            return rc
    binary = work / 'build/custom-format-e2e'
    metadata = {'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                'git_status': subprocess.check_output(['git', 'status', '--short'], cwd=ROOT, text=True),
                'cuda_arch': a.cuda_arch, 'backend_plan': manifest['backend_plan'],
                'operator_sha256': manifest['operator_sha256'],
                'binary_sha256': hashlib.sha256(binary.read_bytes()).hexdigest(),
                'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'gpu': subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version', '--format=csv,noheader'], text=True).strip(),
                'nvcc_version': subprocess.check_output(['nvcc', '--version'], text=True).strip(),
                'cutlass_root': str(a.cutlass_root.resolve()), 'n': a.n, 'warmup': a.warmup,
                'repeat': a.repeat, 'trials': a.trials, 'policy': a.policy, 'autotune_policy': a.autotune_policy,
                'calibration_trials': a.calibration_trials, 'datasets': names,
                'note': 'Existing SM80-compatible generated instructions compiled for selected target; no TMA/WGMMA.'}
    (run / 'run_manifest.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    fields = ['dataset', 'status', 'valid_trials', 'selected_policy', 'selected_task_group_tiles',
              'generated_median_ms', 'cusparse_median_ms', 'speedup_vs_cusparse',
              'max_mismatches', 'max_relative_l2', 'max_normalized_abs']

    def measure(name: str, policy: str, group: int, phase: str, trial: int) -> dict | None:
        result_path = run / f'{name}-{phase}-{trial}.json'
        cmd = [str(binary), '--filename', str((a.matrix_dir / (name + '.mtx')).resolve()),
               '--dataset', name, '--result-json', str(result_path), '--n', str(a.n),
               '--warmup', str(a.warmup), '--repeat', str(a.repeat), '--max-tiles-per-segment', '0',
               '--execution-policy', policy, '--task-group-tiles', str(group)]
        try:
            # Share the project's existing interprocess lock on this host.
            lock_path = Path(os.environ.get('SPARSEKG_GPU_LOCK_FILE', '/tmp/sparsekg_sm80_gpu_benchmark.lock'))
            with lock_path.open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                try:
                    rc = run_logged(cmd, run / f'{name}-{phase}-{trial}.log', timeout=a.timeout)
                finally:
                    fcntl.flock(lock, fcntl.LOCK_UN)
            result = json.loads(result_path.read_text(encoding='utf-8')) if rc == 0 and result_path.is_file() else None
            if result:
                timing, correctness, shape = result['timing'], result['correctness'], result['shape']
                times = (timing['generated_ms'], timing['cusparse_ms'])
                valid = (result['dataset'] == name and shape['N'] == a.n
                         and timing['contract'] == 'sparsene.kernel_only_post_conversion.v1'
                         and timing['warmup'] == a.warmup and timing['repeat'] == a.repeat
                         and correctness['passed'] and correctness['nonfinite'] == 0
                         and all(math.isfinite(t) and t > 0 for t in times))
                if valid:
                    return result
        except (subprocess.TimeoutExpired, OSError, ValueError, KeyError) as exc:
            print(f'{name} {phase} trial {trial}: {exc}', file=sys.stderr, flush=True)
        return None

    failures = 0
    with (run / 'summary.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); f.flush()
        for name in names:
            policy, group = a.policy, a.task_group_tiles
            selection_ok = True
            if a.autotune_policy:
                candidates = []
                base_results = [measure(name, 'auto', a.task_group_tiles, 'cal-baseline', i)
                                for i in range(1, a.calibration_trials + 1)]
                def add_candidate(label: str, selected_policy: str, selected_group: int, items: list) -> None:
                    valid = len(items) == a.calibration_trials and all(x is not None for x in items)
                    candidates.append({'name': label, 'policy': selected_policy, 'group': selected_group,
                                       'valid': valid, 'median_ms': statistics.median(
                                           x['timing']['generated_ms'] for x in items) if valid else None})
                add_candidate('baseline', 'auto', a.task_group_tiles, base_results)
                # Explore only when auto uses a task-mapped heavy-tail path.
                if candidates[0]['valid'] and base_results[0]['packed']['execution_policy'] == 'strict_lb':
                    for candidate_group in (128, 256):
                        if candidate_group == a.task_group_tiles:
                            continue
                        items = [measure(name, 'auto', candidate_group, f'cal-auto-{candidate_group}', i)
                                 for i in range(1, a.calibration_trials + 1)]
                        add_candidate(f'auto-{candidate_group}', 'auto', candidate_group, items)
                    items = [measure(name, 'multi_bind', a.task_group_tiles, 'cal-multi-bind', i)
                             for i in range(1, a.calibration_trials + 1)]
                    add_candidate('multi-bind', 'multi_bind', a.task_group_tiles, items)
                selected = choose_calibrated(candidates)
                (run / f'{name}-selection.json').write_text(json.dumps(
                    {'candidates': candidates, 'selected': selected, 'minimum_gain': 0.015}, indent=2) + '\n', encoding='utf-8')
                if selected is None:
                    selection_ok = False
                else:
                    policy, group = selected['policy'], selected['group']
            values = []
            if selection_ok:
                values = [x for i in range(1, a.trials + 1)
                          if (x := measure(name, policy, group, 'eval', i)) is not None]
            row = {'dataset': name, 'status': 'PASS' if len(values) == a.trials else 'INCOMPLETE',
                   'valid_trials': len(values), 'selected_policy': policy, 'selected_task_group_tiles': group}
            if len(values) == a.trials:
                g = statistics.median(x['timing']['generated_ms'] for x in values)
                c = statistics.median(x['timing']['cusparse_ms'] for x in values)
                row.update(generated_median_ms=g, cusparse_median_ms=c, speedup_vs_cusparse=c/g,
                           max_mismatches=max(x['correctness']['mismatches'] for x in values),
                           max_relative_l2=max(x['correctness']['relative_l2_error'] for x in values),
                           max_normalized_abs=max(x['correctness']['max_abs_normalized'] for x in values))
            else:
                failures += 1
            writer.writerow(row); f.flush()
            print(f"{name}: {row['status']} ({len(values)}/{a.trials}), {policy}/{group}", flush=True)
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
