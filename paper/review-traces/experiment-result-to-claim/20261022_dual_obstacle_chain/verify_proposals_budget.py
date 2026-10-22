"""Neural/proposal and budget algebra only; every simulator entry point is disabled."""
import sys,json,hashlib,zipfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[4];OUT=Path(__file__).resolve().parent;RUN=ROOT/'artifacts/v7_1/dual_obstacle_chain_001';SP=RUN/'static/units/u00'
sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def plain(v):
 if isinstance(v,np.ndarray):return v.tolist()
 if isinstance(v,np.generic):return v.item()
 raise TypeError(type(v).__name__)
def disabled(*a,**k):raise RuntimeError('PHYSICS_AND_GEOMETRY_FORBIDDEN_IN_REVIEW')
import mujoco
for name in ['mj_step','mj_step1','mj_step2','mj_forward','mj_inverse','mj_kinematics','mj_geomDistance','mj_fwdPosition','mj_fwdVelocity','mj_fwdActuation','mj_fwdAcceleration','mj_crb','mj_comPos','mj_collision']:
 if hasattr(mujoco,name):setattr(mujoco,name,disabled)
from ff_prior_static_fullchain_v2.proposals import load_bundle,make
from ff_prior.storage import digest
identity=read(RUN/'run.json')['identity'];task=identity['selection']['entries'][0]['task'];setup_saved=read(SP/'public_setup.json')
setup=(np.array(setup_saved['base']),np.array(setup_saved['jacobian']),np.array(setup_saved['frame']),setup_saved['bow_amplitude_m'])
bundle=load_bundle();neural=[]
for i in range(5):
 a=np.load(SP/f'candidates/{i:02d}/controls.npz');raw,proposal,p=make('diffusion_geometry',i,20264611+i,task,setup,bundle)
 neural.append(dict(id=i,network_calls=p['network_calls'],raw_max_error=float(np.max(abs(raw-a['network_raw']))),proposal_max_error=float(np.max(abs(proposal-a['proposal']))),raw_digest=digest(raw),proposal_digest=digest(proposal)))
# Each Evaluator.predict consumes exactly one coupled call and its interval count.
# Count saved per-grid and sparse linearization evidence in the actual source call order.
budget={};screen_grids=[]
for i in range(5):
 for stage in ['raw','prepared']:
  p=SP/f'candidates/{i:02d}/{stage}/screen.json'
  if p.exists():screen_grids.append(read(p)['intervals'])
budget['screen']=dict(calls=len(screen_grids),intervals=sum(screen_grids))
for i in [4,2]:
 c=SP/f'candidates/{i:02d}';grids=[x['intervals'] for x in read(c/'repair/initial.json')['grid_history']];detail=[]
 for p in sorted((c/'repair/iterations').glob('*/record.json')):
  rec=read(p);a=np.load(p.parent/'linearization.npz');feedback=[x['actual_intervals'] for x in rec['sparse_agreement']];perturb=[len(a['phase'])-1]*int(np.count_nonzero(a['steps']));trials=[x['intervals'] for t in rec['trials'] if t['status']=='EVALUATED' for x in t['grid_history']]
  item=feedback+perturb+trials;grids+=item;detail.append(dict(iteration=rec['iteration'],feedback_calls=len(feedback),perturb_calls=len(perturb),trial_calls=len(trials),calls=len(item),intervals=sum(item)))
 val=read(c/'validated/validation.json');vg=[256]+[v['intervals'] for v in val['validation_grids']];grids+=vg
 budget[f'candidate_{i}']=dict(calls=len(grids),intervals=sum(grids),repair_iterations=len(detail),iteration_detail=detail,validation_calls=len(vg),validation_intervals=sum(vg),calls_cap=3450,intervals_cap=8300000,within_slice=len(grids)<=3450 and sum(grids)<=8300000)
budget['totals']=dict(calls=sum(v['calls'] for v in budget.values()),intervals=sum(v['intervals'] for v in budget.values()))
saved=read(RUN/'static/budget.json');budget['totals']['matches_persisted']=budget['totals']['calls']==saved['actual_coupled']==saved['reserved_coupled'] and budget['totals']['intervals']==saved['integration_intervals']
filters=read(RUN/'dynamic/control_filter.json');filter_report=dict(count=len(filters),nonzero_delta_count=sum(x['delta']>1e-12 for x in filters),maximum_delta=max(x['delta'] for x in filters),all_saved_desired_equal_filtered=all(np.array_equal(x['desired'],x['filtered']) for x in filters),interpretation='The constrained feedback filter is present but inactive on this execution; do not claim an activated online collision correction.')
archive=zipfile.ZipFile(RUN/'executed_source.zip');source_evidence={}
targets={'ff_prior_dual_obstacle_chain/experiment.py':['result=run_unit','terminal=','InitialState','env.advance_native','db.pending','native_steps_cap'], 'ff_prior_dual_obstacle_chain/static.py':['for k,seed','audit_execution','selection='], 'ff_prior_static_goal_seed/tracker.py':['robot.calls','mj_step','qfrc_applied','data.ctrl'], 'ff_prior_static_goal_seed/physical_audit.py':['initial.reset','mj_step','physics_steps','validate(','collision.scan'], 'ff_prior_terminal_tracking/environment.py':['InitialState','self.path','self.pairs','mj_step','reference_q'], 'ff_prior_dual_obstacle_chain/evaluation.py':['initial.reset','mj_step','fullerr','no_hidden_reset'], 'ff_prior/robot_adapter.py':['for obstacle','worldbody.add_geom','contype=1','scene_model_hash'], 'ff_prior/geometry.py':['self.geoms','self.pairs.append','module_neighbours','intentional_contact'], 'ff_prior_dual_obstacle_chain/visuals.py':['geomgroup','update_scene','obstacles=','source_static_hash']}
for name,tokens in targets.items():source_evidence[name]=[dict(line=i,text=line) for i,line in enumerate(archive.read(name).decode('utf-8-sig').splitlines(),1) if any(t in line for t in tokens)]
out=dict(new_physics_steps=0,new_coupled_calls=0,new_geometry_queries=0,neural=neural,network_raw_bit_exact=all(r['network_calls']==16 and r['raw_max_error']==0 for r in neural),public_bow_proposal_atol_1e12=all(r['proposal_max_error']<=1e-12 for r in neural),public_bow_note='Public bow is a fresh small linear solve; this review observes up to 1.01e-13 rad arithmetic discrepancy and does not label it bit exact.',budget=budget,filter=filter_report,source_readback=source_evidence)
(OUT/'proposal_budget_evidence.json').write_text(json.dumps(out,indent=2,ensure_ascii=False,default=plain)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in out.items() if k!='source_readback'},indent=2,default=plain))
