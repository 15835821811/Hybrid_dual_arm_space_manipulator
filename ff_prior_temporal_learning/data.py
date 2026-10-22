import numpy as np
from tracking_diffusion.environment import TrackingEnv,Task
from ff_prior.robot_adapter import RobotAdapter
from ff_prior.scenes import SceneSpec
from ff_prior.storage import read_json,digest,file_digest
from ff_prior_dynamic_reference.execution import Environment
from ff_prior_dynamic_handoff.execution import Environment as DevEnvironment,task_identity
from .protocol import ROOT,DATA,DEV
from .features import public_features,fit_codec,encode_conditions,encode_reference

def source_environment(row):
    ident=read_json(ROOT/row['identity'])
    old=TrackingEnv(Task(**ident['task']));scene=SceneSpec.from_dict(ident['scene']);robot=RobotAdapter(scene=scene)
    env=Environment(old,robot)
    with np.load(ROOT/row['initial'],allow_pickle=False) as a:np.testing.assert_array_equal(env.initial.values,a['values'])
    assert env.initial.state_id==ident['initial_state_id'] and digest(robot.identity())==digest(ident['robot'])
    return env,scene

def development_environment(scene):
    robot=RobotAdapter();env=DevEnvironment(robot,scene=='drifting',20261021 if scene=='drifting' else 20261020)
    original=DEV/('u06' if scene=='drifting' else 'u00')
    assert digest(task_identity(env,scene,20261021 if scene=='drifting' else 20261020))==digest(read_json(original/'task.json'))
    with np.load(original/'initial.npz',allow_pickle=False) as a:np.testing.assert_array_equal(env.initial.values,a['values'])
    return env,SceneSpec()

def load_data(progress):
    view=read_json(DATA/'qualified_view.json');assert view['gate']['passed']
    rows=view['rows'];assert len(rows)==12
    all_features=[];coefficients=[]
    for row in rows:
        assert row['grade']=='EXECUTION_VERIFIED'
        assert file_digest(ROOT/row['reference'])==row['reference_sha256']
        env,scene=source_environment(row);features=public_features(env,scene)
        with np.load(ROOT/row['reference'],allow_pickle=False) as a:
            coefficients.append(a['coefficients'].copy())
            np.testing.assert_array_equal(features['knots'],a['knots'])
        all_features.append(features);progress('public_condition_construction',unit=row['unit'])
        del env
    train=[i for i,r in enumerate(rows) if r['split']=='train'];val=[i for i,r in enumerate(rows) if r['split']=='validation']
    assert len(train)==8 and len(val)==4
    codec=fit_codec([coefficients[i] for i in train],[all_features[i] for i in train])
    arrays={}
    for split,indices in [('train',train),('validation',val)]:
        pairs=[encode_conditions(all_features[i],codec) for i in indices]
        arrays[split]=dict(x=np.stack([encode_reference(coefficients[i],all_features[i],codec) for i in indices]),
            global_condition=np.stack([p[0] for p in pairs]),local_condition=np.stack([p[1] for p in pairs]),
            split=[split]*len(indices),units=[rows[i]['unit'] for i in indices])
    return codec,arrays,rows,all_features
