"""Post-hoc saved-pose chain-rule sensitivity; no new propagation, no gate reclassification."""
from pathlib import Path
import json,hashlib
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[4];OUT=Path(__file__).resolve().parent;RUN=ROOT/'artifacts/v7_1/smooth_endpoint_bounded_001'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
idx=read(ROOT/'artifacts/v7_1/expand256_train_dev_001/training_input_audit.json')['records']
rows={e['sample_id']:e for e in idx}
def qgroups(ee,goal):
 a,b=np.asarray(ee),np.asarray(goal);p=np.linalg.norm(a[...,:3,3]-b[...,:3,3],axis=-1);r=Rotation.from_matrix(np.swapaxes(b[...,:3,:3],-1,-2)@a[...,:3,:3]).magnitude()
 return np.r_[(p/.005)**2,(r/np.deg2rad(1))**2]
result=[]
for case in [8,12]:
 d=RUN/'diagnostic'/f'{case:02d}';proof=read(d/'gradient_1.json');a=np.load(d/'gradient_1.npz');goal=read(ROOT/rows[proof['sample_id']]['record'])['task']['condition']['goals']
 q=qgroups(proof['central']['ee'],goal);soft=np.exp(q-q.max());soft/=soft.sum();w=soft+.0125
 pert=np.array([[qgroups(ee,goal) for ee in pair] for pair in a['perturbation_ee']]);J=(pert[:,1]-pert[:,0])/(2*proof['epsilon']);gchain=J@w;direction=np.array(proof['direction']);chain_dot=float(gchain@direction)
 qdirs=np.array([qgroups(s['ee'],goal) for s in proof['direction_samples']]);chain_dir=float((qdirs[1]-qdirs[0])/(2*proof['epsilon'])@w);raw=proof['direction_slope'];parts=a['gradient']*direction
 result.append({'case':case,'original_relative_direction_error':proof['direction_relative_error'],'original_scalar_direction_slope':raw,'original_gradient_dot_direction':proof['direction_expected'],'sum_abs_projected_coordinate_terms':float(abs(parts).sum()),'posthoc_chain_gradient_dot_direction':chain_dot,'posthoc_chain_direct_q_slope':chain_dir,'chain_dot_vs_original_scalar_slope_relative_error':abs(chain_dot-raw)/max(abs(chain_dot),abs(raw),1e-6),'internal_chain_direction_relative_error':abs(chain_dot-chain_dir)/max(abs(chain_dot),abs(chain_dir),1e-6),'changes_frozen_verdict':False})
(OUT/'saved_pose_chain_sensitivity.json').write_text(json.dumps({'status':'posthoc_diagnostic_only','no_new_physics':True,'no_replacement_gradient_used_for_training':True,'original_gate_preserved':'46/48; M1 BLOCKED','results':result},ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
