"""Validate one generated scalar-compact executable over the 26 matrices.

Usage: python run_scalar_compact_26_validation.py EXECUTABLE NEW_OUTPUT_DIR
GPU executions are serialized with the shared Sparsene benchmark lock.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time

DATASETS = Path('/mnt/lustre-client/zhangkaige/selected_mtx')
LOCK = '/tmp/sparsekg_sm80_gpu_benchmark.lock'


def nnz(path: Path) -> int:
    with path.open() as source:
        for line in source:
            if not line.startswith('%'):
                return int(line.split()[2])
    raise ValueError(f'missing MatrixMarket dimensions: {path}')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('executable', type=Path)
    parser.add_argument('output_dir', type=Path)
    parser.add_argument('--n', type=int, default=512)
    parser.add_argument('--timeout', type=int, default=600)
    args = parser.parse_args()
    executable = args.executable.resolve(strict=True)
    cases = sorted(DATASETS.glob('*.mtx'), key=nnz)
    if len(cases) != 26:
        raise RuntimeError(f'expected 26 matrices, found {len(cases)} in {DATASETS}')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    reports = []
    for index, matrix in enumerate(cases, 1):
        result = args.output_dir / f'{matrix.stem}.json'
        command = ['flock', '-F', LOCK, str(executable), '--filename', str(matrix),
                   '--dataset', matrix.stem, '--result-json', str(result),
                   '--n', str(args.n), '--warmup', '2', '--repeat', '5',
                   '--execution-policy', 'row']
        start = time.monotonic()
        try:
            process = subprocess.run(command, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, text=True,
                                     timeout=args.timeout, check=False)
            output, code = process.stdout, process.returncode
        except subprocess.TimeoutExpired as exc:
            output = exc.stdout or ''
            if isinstance(output, bytes):
                output = output.decode(errors='replace')
            output += '\nTimed out.\n'
            code = 124
        (args.output_dir / f'{matrix.stem}.log').write_text(output)
        item = {'dataset': matrix.stem, 'exit_code': code,
                'seconds': time.monotonic() - start}
        if result.exists():
            payload = json.loads(result.read_text())
            item['correctness'] = payload.get('correctness')
            item['timing'] = payload.get('timing')
        reports.append(item)
        (args.output_dir / 'summary.json').write_text(json.dumps(reports, indent=2) + '\n')
        print(f'{index}/{len(cases)} {matrix.stem}: exit={code}', flush=True)
    return 0 if all(item['exit_code'] == 0 and item.get('correctness', {}).get('passed') is True for item in reports) else 1


if __name__ == '__main__':
    raise SystemExit(main())
