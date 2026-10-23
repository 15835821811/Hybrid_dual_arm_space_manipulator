"""Model-parameter bound only; no state sampling or physics advance."""
import json
import sys
from pathlib import Path
import numpy as np
import mujoco

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

def main():
    def forbidden(*args, **kwargs):
        raise RuntimeError('READ_ONLY_MODEL_PARAMETERS_NO_PHYSICS')
    for name in ('mj_step', 'mj_step1', 'mj_step2', 'mj_forward',
                 'mj_Euler', 'mj_implicit', 'mj_RungeKutta', 'mj_integratePos'):
        setattr(mujoco, name, forbidden)
    from ff_prior.robot_adapter import RobotAdapter
    from ff_prior.scenes import SceneSpec
    contract = json.loads((ROOT/'ff_prior_satellite_origin/evidence/contract.json').read_text(encoding='utf-8-sig'))
    r = RobotAdapter(scene=SceneSpec.from_dict(contract['scene']))
    m = r.model
    damped = m.dof_damping > 0
    covered = bool(np.all(m.dof_armature[damped] > 0))
    rho = float(np.max(m.dof_damping[damped]/m.dof_armature[damped])) if covered else None
    out = {
        'kind': 'MODEL_PARAMETER_INEQUALITY_NOT_TRAJECTORY_VALIDATION',
        'mujoco_version': mujoco.__version__,
        'scene_model_hash': r.identity()['scene_model_hash'],
        'all_damped_dofs_have_positive_armature': covered,
        'damped_dofs': np.flatnonzero(damped).tolist(),
        'damping': m.dof_damping.tolist(), 'armature': m.dof_armature.tolist(),
        'base_frictionloss': m.dof_frictionloss[r.base_dofs].tolist(),
        'global_passive_damping_rate_bound_per_s': rho,
        'h_times_bound': None if rho is None else float(m.opt.timestep*rho),
        'derivation': 'M(q) >= diag(armature); D <= rho*diag(armature), so lambda_max(M^-1 D) <= rho. This bounds only joint passive damping, not the sampled controller or nonlinear dynamics.',
        'rk4_negative_real_stability_endpoint_approx': 2.785,
        'rk4_scalar_passive_damping_condition': covered and m.opt.timestep*rho < 2.785,
        'trajectory_qualification': 'NOT_RUN',
        'new_native_steps': 0, 'new_coupled_calls': 0,
        'new_static_mass_queries': 0, 'new_geometry_queries': 0,
        'earlier_initial_mass_query': {'count': 1, 'maximum_generalized_damping_eigenvalue_per_s': 4.999413884480664, 'role': 'diagnostic_only_not_needed_for_global_bound'},
    }
    (ROOT/'ff_prior_momentum_diagnosis/evidence/passive_bound.json').write_text(json.dumps(out, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k:v for k,v in out.items() if k not in ('damping','armature','damped_dofs')}, ensure_ascii=False))

if __name__ == '__main__':
    main()
