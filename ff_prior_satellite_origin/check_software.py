"""Exact handoff/force algebra/interface checks, with native stepping disabled."""
import inspect,time
import numpy as np
import mujoco
from unittest.mock import patch
from ff_prior.storage import read_json,digest
from ff_prior_pilot_train.storage import atomic_json
from ff_prior.robot_adapter import RobotAdapter
from ff_prior.scenes import SceneSpec
from ff_prior_dynamic_reference.execution import integration
from ff_prior.coupled_predictor import pose_errors
from .protocol import ROOT,PACKAGE,OUT,P
from .inputs import sources
from .environment import Environment
from .force import force_at,analytic_motion,satellite_state

def main():
    started=time.time();assert not OUT.exists();checks=[]
    c=read_json(PACKAGE/'evidence/contract.json');t=dict(np.load(PACKAGE/'evidence/terminal.npz'))
    with patch.object(mujoco,'mj_step',side_effect=AssertionError('PREFLIGHT_PHYSICS_FORBIDDEN')):
        r=RobotAdapter(scene=SceneSpec.from_dict(c['scene']))
        env=Environment(r,t,c,None,record_initial=False)
        np.testing.assert_array_equal(integration(r,env.data),t['integration']);assert len(t['integration'])==836
        assert abs(env.data.time-54.5)<1e-8;checks.append('exact_836_54_5s_handoff_without_native_step')
        np.testing.assert_array_equal(env.command,t['reference_qd'])
        np.testing.assert_array_equal(env.reference_q,t['reference_q']);checks.append('controller_memory_exact')
        np.testing.assert_allclose(env.desired_poses(env.origin_time),c['initial_desired'],atol=1e-14)
        pos,rot=pose_errors(r.ee(env.data),env.desired_poses(env.origin_time))
        assert np.all(pos<.005) and np.all(rot<np.pi/180);checks.append('continuous_reference_and_original_initial_error_gates')
        for dt in (0.,2.,10.,29.):
            np.testing.assert_array_equal(env.reference(env.origin_time+dt)[0],np.asarray(c['hold_continuum_pose'])[:3,3])
        checks.append('W7_world_goal_held_no_waypoint_restart')
        assert np.count_nonzero(env.data.xfrc_applied)==np.count_nonzero(env.data.qfrc_applied)==0
        assert np.count_nonzero(env.data.qvel[env.tv])==0;checks.append('zero_inherited_satellite_velocity_and_external_inputs')
        assert r.model.body_mass[env.target_id]==20.
        np.testing.assert_array_equal(r.model.body_ipos[env.target_id],np.zeros(3))
        assert np.all(r.model.opt.gravity==0);checks.append('same_model_mass_COM_and_zero_gravity')
        state=satellite_state(r,env.data,env.target_id);assert state.shape==(25,) and np.count_nonzero(state[12:])==0
        checks.append('world_COM_velocity_momentum_energy_interface')
        forces=np.asarray([force_at((i+.5)*.002) for i in range(14500)])
        np.testing.assert_allclose(forces.sum(axis=0)*.002,[0,0,.04],atol=1e-14)
        assert np.count_nonzero(forces[1000:])==0 and np.max(forces)<=.04;checks.append('discrete_force_impulse_and_exact_coast_shutdown')
        z,v=analytic_motion(np.array([0.,2.,29.]),20.)
        np.testing.assert_allclose(z,[0,.002,.056],atol=1e-14);np.testing.assert_allclose(v,[0,.002,.002],atol=1e-14)
        checks.append('independent_analytic_design_values')
        # Conservative swept AABB contains the satellite cube for this zero-spin pulse.
        center=state[:3];half=np.abs(state[3:12].reshape(3,3))@np.full(3,.2)
        lower=center-half;upper=center+half+np.array([0,0,.056003])
        clearance=[]
        for obj in c['scene']['objects']:
            point=np.asarray(obj['position']);delta=np.maximum(np.maximum(lower-point,point-upper),0)
            clearance.append(float(np.linalg.norm(delta)-obj['size'][0]))
        assert min(clearance)>=0;checks.append('analytic_satellite_sweep_disjoint_from_all_seven_fixed_spheres')
        # This is not a model path certificate for moving robot geometry.
        source=inspect.getsource(Environment.advance_native)
        for term in ('qpos[','qvel['):
            for line in source.splitlines():
                if term in line:assert line.strip().startswith(('aref=','r.forward','ba=','raw=','force=')) or '=' not in line.split(term)[0]
        assert 'mj_setState' not in source and 'mj_resetData' not in source
        checks.append('actual_loop_only_robot_controls_and_disclosed_force_writes')
        assert P['max_native_replay_steps']==29000==P['steps_each']*2
        assert not(P['training'] or P['expansion'] or P['TEST'] or P['advantage_comparison'])
        checks.append('one_candidate_budget_and_scope')
    result=dict(status='PASS',terminal_preflight=True,checks=checks,tests=len(checks),source_hashes=sources(),
        terminal_digest=digest(t['integration']),new_native_steps=0,new_coupled_calls=0,new_neural_samples=0,
        new_geometry_queries=0,elapsed_s=time.time()-started,satellite_sweep_sphere_clearance_lower_m=clearance,
        model_qualification='input identity and analytic satellite origin only; no new whole-robot MODEL_VALID trajectory claim')
    atomic_json(PACKAGE/'evidence/software_tests.json',result)
    print({k:v for k,v in result.items() if k!='source_hashes'})
if __name__=='__main__':main()
