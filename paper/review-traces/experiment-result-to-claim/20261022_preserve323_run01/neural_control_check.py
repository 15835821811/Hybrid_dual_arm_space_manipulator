"""Saved raw control binding and local sensitivity diagnostic; no physics/update."""
import sys
sys.dont_write_bytecode=True
import pathlib,json,numpy as np,torch
ROOT=pathlib.Path(__file__).resolve().parents[4];TRACE=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT));torch.set_num_threads(2)
from ff_prior_preserve323.model import TransferDenoiser
from ff_prior_preserve323.codec import FullCodec
from ff_prior_control_transfer.model import rollout
R=ROOT/'artifacts/v7_1/preserve323_bounded_001'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
load=lambda p:torch.load(p,map_location='cpu',weights_only=False)
sel=read(R/'frozen_tasks.json');codec=FullCodec(read(R/'codec.json'))
audit=read(ROOT/'artifacts/v7_1/expand256_train_dev_001/training_input_audit.json');ids={x['sample_id']:i for i,x in enumerate(audit['records'])}
with np.load(ROOT/'artifacts/v7_1/expand256_train_dev_001/loaded_training_arrays.npz') as a:bases=a['bases'].copy()
w0=load(R/'TRAIN_fit/initial.pt')['weights'];wp=load(R/'TRAIN_fit/rounds/00/proposed.pt')['weights'];model=TransferDenoiser(codec,warm=False)
groups=[('TRAIN_baseline/fit','fit',0),('TRAIN_baseline/check','check',0),('TRAIN_check_final','check',0)]+[(f'TRAIN_fit/rounds/00/trial_{i}','fit',s) for i,s in enumerate([1.,.5,.25])]
output={}
with torch.no_grad():
 for path,role,scale in groups:
  model.load_state_dict({k:w0[k]+scale*(wp[k]-w0[k]) if k.startswith('detail.') else w0[k] for k in w0})
  errors=[]
  for j,(row,seeds) in enumerate(zip(sel[role+'_pool'],sel[role+'_seed_pairs'])):
   base=bases[ids[row['sample_id']]];center=codec.center(row['condition'],base);c=torch.tensor(codec.condition(row['condition'],base)[None],dtype=torch.float32)
   for k,seed in enumerate(seeds):
    full=rollout(model,c,[seed])[0].numpy();raw=codec.decode(codec.coefficients(full,center),base)
    with np.load(R/path/f'{2*j+k:02d}'/'controls.npz') as a:errors.append(float(np.max(np.abs(raw-a['raw']))))
  output[path]={'n':len(errors),'recomputed_raw_control_max_error':max(errors)}
  assert max(errors)==0
ratios=[]
for j in range(32):
 with np.load(R/f'TRAIN_fit/rounds/00/jacobians/{j:02d}.npz') as a:
  ratios.append(float(np.linalg.norm(a['J'][:,102:])/np.linalg.norm(a['J'][:,:102])))
output['local_sensitivity']={'detail_to_low_Frobenius_ratio_min':min(ratios),'detail_to_low_Frobenius_ratio_median':float(np.median(ratios)),'detail_to_low_Frobenius_ratio_max':max(ratios),'interpretation':'Local normalized-coordinate sensitivity only; weak detail endpoint authority is a hypothesis, not an impossibility proof or isolated causal claim.'}
output['status']='PASS'
(TRACE/'neural_control_check.json').write_text(json.dumps(output,indent=2),encoding='utf-8');print(json.dumps(output,indent=2))
