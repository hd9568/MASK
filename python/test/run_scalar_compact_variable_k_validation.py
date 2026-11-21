"""Generate, compile and validate the exact four-level backend without an agent.

Example (activate the project environment and set PYTHONPATH=python:python/test):
  python python/test/run_scalar_compact_variable_k_validation.py /lustre/new-run \
      --outer-k 21 --inner-k 7 --matrix /path/to/matrix.mtx

Use SPARSEKG_CUTLASS_ROOT and SPARSEKG_CUDA_ARCH for an external CUTLASS or a
native H100 build. Output must be a new directory; no existing run is deleted.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import random
from pathlib import Path

from generate_custom_format_operators import generate_one
from run_custom_format_e2e import compile_one, run_one


def synthetic(path: Path) -> None:
    rng = random.Random(104)
    coords = {(r, c) for r in list(range(15)) + list(range(32, 39))
              for c in range(73) if rng.random() < 0.19}
    lines = ['%%MatrixMarket matrix coordinate real general',
             '% partial rows and empty window', f'39 73 {len(coords)}']
    lines += [f'{r+1} {c+1} {((r*13+c*7)%17-8)/8:.3f}'
              for r, c in sorted(coords)]
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('output_dir', type=Path)
    p.add_argument('--outer-k', type=int, required=True)
    p.add_argument('--inner-k', type=int, required=True)
    p.add_argument('--matrix', type=Path, action='append', default=[])
    p.add_argument('--n', type=int, default=64)
    p.add_argument('--jobs', type=int, default=4)
    args = p.parse_args()
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root/'formats').mkdir()
    dataset = 'scalar_compact_validation'
    data = root/'datasets'/dataset
    data.mkdir(parents=True)
    fixture = Path(__file__).parent/'fixtures/historical_webbase_four_level.json'
    spec = json.loads(fixture.read_text(encoding='utf-8'))
    spec['dataset'] = dataset
    spec['params']['BLK_K'] = args.outer_k
    spec['params']['BLK_K_I'] = args.inner_k
    (root/'formats'/f'{dataset}.format.json').write_text(json.dumps(spec, indent=2))
    small = root/'small.mtx'
    synthetic(small)
    manifest = generate_one(root, dataset, require_exact_backend=True)
    binary = compile_one(root, manifest, args.jobs)
    reports = []
    for idx, matrix in enumerate([small, *args.matrix]):
        matrix = matrix.resolve(strict=True)
        (data/'matrix_features.json').write_text(json.dumps({'path': str(matrix)}))
        with open('/tmp/sparsekg_sm80_gpu_benchmark.lock', 'w') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            result = run_one(root, manifest, binary, n=args.n, warmup=2, repeat=5,
                             execution_policy='row', result_label=f'case{idx}')
        reports.append({'matrix': str(matrix), 'correctness': result['correctness'],
                        'timing': result['timing'], 'result_path': result['result_path']})
        (root/'validation_summary.json').write_text(json.dumps(reports, indent=2))
    return 0 if all(r['correctness']['passed'] and not r['correctness']['mismatches']
                    and not r['correctness']['nonfinite'] for r in reports) else 1


if __name__ == '__main__':
    raise SystemExit(main())
