from pathlib import Path
import argparse
import numpy as np
from scipy.io import mmread
parser=argparse.ArgumentParser()
parser.add_argument('--matrix',type=Path,required=True)
parser.add_argument('--data',type=Path,required=True)
args=parser.parse_args()
base=args.data
for leaf in ('coo','dense'):
 for p in (base/leaf).glob('*.npy'):
  a=np.load(p,mmap_mode='r'); a.tofile(base/leaf/(p.stem+'.bin'))
  print(leaf,p.stem,a.shape,a.dtype,flush=True)
a=mmread(args.matrix).astype(np.float32).tocsr();a.sum_duplicates();a.eliminate_zeros();a.sort_indices()
for n,arr in (('csr_ptr',a.indptr.astype(np.int32)),('csr_col',a.indices.astype(np.int32)),('csr_val',a.data)):
 arr.tofile(base/(n+'.bin'));print(n,len(arr),flush=True)
