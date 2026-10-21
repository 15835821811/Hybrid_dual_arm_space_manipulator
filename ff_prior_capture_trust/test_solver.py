"""Software regression checks, not robot qualification."""
import numpy as np
import pytest
from ff_prior.storage import read_json
from ff_prior_latent.learning import Codec
from ff_prior_capture_feasibility.solver import basis
from ff_prior_pilot_train.storage import atomic_json
from .solver import solve_step,accept,restore_feasible,trust_decision,active_merit
from .experiment import assert_resume_safe
from .protocol import P,SOURCE,PREVIOUS

def record(q,d):
 e=np.zeros(12);e[::3]=q
 return {'goal':e,'minima':np.array(d)}

def test_all_physical_goal_components_protected():
 assert not accept(record([.89,.2,.1,.1],[-.04]),record([1.01,.2,.1,.1],[-.001]))
 assert not accept(record([2.,.2,.1,.1],[-.04]),record([1.5,1.1,.1,.1],[-.001]))
 assert accept(record([.89,.2,.1,.1],[-.04]),record([.99,.2,.1,.1],[-.03]))

def test_zero_feasible_regression_on_original_00_linearization():
 with np.load(PREVIOUS/'teachers/00_original_failed/iterations/00/linearization.npz') as a:
  z,e,d,jg,jd=[a[k].copy() for k in ('Z','goal','active_distances','jg','jd')]
 codec=Codec(read_json(SOURCE/'learning/aligned/codec.json'))
 # Broad test-only limits avoid model construction; actual physical limits are used by worker.
 step,r=solve_step(z,basis('full323',codec),e,d,jg,jd,-np.full(17,4.),np.full(17,4.))
 assert r['initialization']=='ZERO_FEASIBLE' and r['returned_feasible']
 assert r['initial_residual']['pose_min']>=0 and abs(step).max()<=.15+1e-10
 assert r['returned_objective']<=r['initial_objective']+1e-12
 assert 'solved_u' in r and 'solved_residual' in r

@pytest.mark.parametrize('magnification',[1.,3.])
def test_actual_control_step_scale(magnification):
 z=np.zeros((19,17));D=np.zeros((2,19,17));D[0,0,0]=magnification;D[1,1,1]=magnification
 G=np.zeros((12,2));J=np.array([[magnification,0.]])
 step,r=solve_step(z,D,np.zeros(12),np.array([-.4]),G,J,-np.ones(17),np.ones(17))
 delta=np.einsum('i,ijk->jk',step,D)
 assert r['returned_feasible'] and abs(delta).max()<=.15+1e-8
 assert delta[0,0]>.1

def test_codec_controls_domain_and_goal():
 codec=Codec(read_json(SOURCE/'learning/aligned/codec.json'));D=basis('codec102',codec)
 z=np.zeros((19,17));G=np.zeros((12,102));G[0,0]=1.
 J=D[:,4,11][None,:]
 step,r=solve_step(z,D,np.zeros(12),np.array([-.03]),G,J,-np.full(17,.04),np.full(17,.04))
 delta=np.einsum('i,ijk->jk',step,D)
 assert abs(delta).max()<=.04+1e-8 and abs((G@step)[0])<=1+1e-8
 assert r['returned_feasible']

def test_unknown_work_not_repeated(tmp_path):
 atomic_json(tmp_path/'ledger.json',{'inflight':{'id':'unknown'}})
 with pytest.raises(ValueError,match='UNKNOWN_WORK'):assert_resume_safe(tmp_path)

def test_budgets_and_no_dataset_mutation():
 assert P['seeds']==2 and not P['TEST'] and not P['training'] and P['TRAIN_added']==0
 assert 2*12*323+8*102+500<P['coupled_cap']
 assert P['validation_intervals'][-1]==16384

def test_real_saved_infeasible_00_has_nonzero_ray_recovery():
 r=read_json(PREVIOUS/'teachers/00_original_failed/iterations/00/record.json')['solver']
 lim=read_json(PREVIOUS/'physical_limits.json')
 with np.load(PREVIOUS/'teachers/00_original_failed/iterations/00/linearization.npz') as a:
  z=a['Z'];e=a['goal'];G=a['jg'];u=np.asarray(r['solved_u'])
  def feasible(v):
   delta=(.15*v).reshape(z.shape)
   return (max(abs(delta).ravel())<=.15+1e-8 and np.all(z+delta>=np.asarray(lim['lower'])-1e-8)
       and np.all(z+delta<=np.asarray(lim['upper'])+1e-8)
       and max(np.sum((e+G@(.15*v)).reshape(4,3)**2,axis=1))<=1+1e-8)
  assert not feasible(u)
  restored,alpha=restore_feasible(u,feasible)
  assert 0<alpha<1 and feasible(restored) and abs(restored).max()>0

def test_trust_rejection_relinearizes_at_smaller_radius_and_is_bounded():
 r=P['trust_initial_rad']
 assert trust_decision(r,False)==r/2
 assert trust_decision(r,True,.5,1.,True)==r
 assert trust_decision(r,True,.9,1.,True)==r*2
 assert trust_decision(r,True,.9,.125,True)==r/2
 assert trust_decision(P['trust_min_rad'],False)==P['trust_min_rad']
 assert trust_decision(P['trust_rad'],True,.9,1.,True)==P['trust_rad']

def test_ratio_compares_identical_active_pairs_not_new_selection():
 e=np.zeros(12);d=np.array([-.004,-.002]);delta=np.array([.001,.0005])
 predicted=active_merit(e,d)-active_merit(e,d+delta)
 actual=active_merit(e,d)-active_merit(e,d+delta/2)
 assert predicted>actual>0 and actual/predicted>P['ratio_accept']
 assert active_merit(e,np.array([]))==0
