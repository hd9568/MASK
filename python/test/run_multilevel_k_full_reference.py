from pathlib import Path
import argparse,json,time
import numpy as np
from scipy import sparse
from scipy.io import mmread
from sparsene.op_gen.multilevel_k_lowering import lower_multilevel_k_spec
from sparsene.formats.multilevel_k_reference import pack_multilevel_k
parser=argparse.ArgumentParser()
parser.add_argument('--matrix',type=Path,required=True)
parser.add_argument('--spec',type=Path,required=True)
parser.add_argument('--out',type=Path,required=True)
args=parser.parse_args()
root=args.out
root.mkdir(parents=True,exist_ok=True)
spec=json.loads(args.spec.read_text())
start=time.monotonic()
a=mmread(args.matrix).astype(np.float32).tocsr(); a.sum_duplicates();a.eliminate_zeros();a.sort_indices()
print('read',a.shape,a.nnz,'seconds',round(time.monotonic()-start,2),flush=True)
for leaf in ('coo','dense'):
 spec['atomic']['type']=leaf;plan=lower_multilevel_k_spec(spec); t=time.monotonic()
 p=pack_multilevel_k(a,blk_m=plan.blk_m,widths=plan.widths,leaf=leaf)
 if len(plan.widths)!=2: raise ValueError('full-matrix vectorized verifier currently supports one inner level')
 print('packed',leaf,'seconds',round(time.monotonic()-t,2),'roots',len(p.root_column_ids),'leaves',len(p.child_indices[0]),flush=True)
 # Vectorized independent reconstruction (not unpack_multilevel_k).
 root_to_window=np.repeat(np.arange(len(p.window_root_ptr)-1),np.diff(p.window_root_ptr))
 leaf_to_root=np.repeat(np.arange(len(p.root_column_ids)),np.diff(p.child_ptrs[0]))
 if leaf=='coo':
  item_to_leaf=np.repeat(np.arange(len(p.child_indices[0])),np.diff(p.leaf_value_ptr))
  rr=p.local_rows;cc=p.local_cols;vv=p.values
 else:
  item_to_leaf,rr,cc=np.nonzero(p.values);vv=p.values[item_to_leaf,rr,cc]
 item_to_root=leaf_to_root[item_to_leaf]
 row=root_to_window[item_to_root]*p.blk_m+rr
 col=p.root_column_ids[item_to_root,p.child_indices[0][item_to_leaf]*p.widths[-1]+cc]
 assert np.all(row<a.shape[0]) and np.all(col>=0) and np.all(col<a.shape[1])
 restored=sparse.coo_matrix((vv,(row,col)),shape=a.shape).tocsr();restored.sum_duplicates();restored.sort_indices()
 assert np.array_equal(restored.indptr,a.indptr) and np.array_equal(restored.indices,a.indices) and np.array_equal(restored.data,a.data)
 print('PASS full structure and values',leaf,'nnz',len(vv),'seconds',round(time.monotonic()-t,2),flush=True)
 arrays={'window_root_ptr':p.window_root_ptr,'root_column_ids':p.root_column_ids,'child_ptr0':p.child_ptrs[0],'child_index0':p.child_indices[0],'values':p.values}
 if leaf=='coo': arrays.update(value_ptr=p.leaf_value_ptr,local_rows=p.local_rows,local_cols=p.local_cols)
 d=root/leaf;d.mkdir(exist_ok=True)
 for name,arr in arrays.items(): np.save(d/(name+'.npy'),arr,allow_pickle=False)
 (d/'manifest.json').write_text(json.dumps({'shape':a.shape,'nnz':a.nnz,'plan':plan.to_dict(),'arrays':{n:{'shape':v.shape,'dtype':str(v.dtype)} for n,v in arrays.items()}},indent=2)+'\n')
 print('saved',d,flush=True)
print('done seconds',round(time.monotonic()-start,2),flush=True)
