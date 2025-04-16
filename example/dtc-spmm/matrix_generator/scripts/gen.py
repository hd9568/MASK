import os
import pathlib


def gen_matrix(M, K, sparsity):
    script_path = pathlib.Path(__file__).parent.resolve()
    gen_exe_path = script_path.parent / "build" / "gen"
    out_dir = script_path.parent / "generated_matrices"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{M}_{K}_{sparsity}.mtx"
    os.system(f"{gen_exe_path} -M {M} -K {K} -sparsity {sparsity} -filename {out_path}")


if __name__ == "__main__":
	for m, k in [(512, 512), (1024, 1024), (2048, 2048), (4096, 4096)]:
    # for m in [512, 1024, 2048, 4096]:
    #     for k in [512, 1024, 2048, 4096]:
		for sp in [0.5, 0.75, 0.8, 0.9, 0.95]:
			gen_matrix(m, k, sp)
