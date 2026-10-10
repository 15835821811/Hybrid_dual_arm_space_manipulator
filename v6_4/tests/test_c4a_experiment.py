"""C4 DEV budget/isolation/phase-seal tests without any physical calls."""
import copy
from pathlib import Path
from unittest.mock import patch

import mujoco
import numpy as np
import pytest

from v6_4.c4a_experiment import reserve,dev_noise,seal_dev,LIMITS
from v6_4.c4a_frozen import RELEASE
from v6_4.route_optimizer_protocol import TaskSpec,read,write,sha
from v6_4.closed_loop_warmstart_validation import verify_phase_selections,actual_alias_identity


def test_budget_reservation_hard_caps_and_no_duplicate_retry(tmp_path):
    for i in range(6): reserve(tmp_path,'candidate_slots',str(i),8)
    with pytest.raises(ValueError,match='hard budget'): reserve(tmp_path,'candidate_slots','extra',1)
    with pytest.raises(RuntimeError,match='no implicit'): reserve(tmp_path,'candidate_slots','0',8)
    for i in range(2): reserve(tmp_path,'actual_logical_slots',str(i),6)
    with pytest.raises(ValueError): reserve(tmp_path,'actual_logical_slots','extra',1)
    for i in range(4): reserve(tmp_path,'ddim_samples',str(i),2)
    with pytest.raises(ValueError): reserve(tmp_path,'ddim_samples','extra',1)
    assert LIMITS==dict(candidate_slots=48,actual_logical_slots=12,ddim_samples=8)


def test_dev_noise_deterministic_task_bound_and_separate_seed():
    from v6_4.simple_warmstart_regression import frozen_noise
    task=TaskSpec.from_dict(read(RELEASE/'snapshot/frozen_tasks/c1_mother_00_plus/task.json'))
    a=dev_noise(task);b=dev_noise(task);c=frozen_noise(task,64324)
    assert set(a)=={1,3}
    for k in a:
        np.testing.assert_array_equal(a[k],b[k]);assert a[k].dtype==np.float32 and not np.array_equal(a[k],c[k])


def test_dev_complete_phase_seal_and_tamper_rejection(tmp_path):
    rows=[dict(task_id='dev'+str(i),task_sha256='sha'+str(i)) for i in range(2)]
    plan=dict(tasks=rows);write(tmp_path/'plan.json',plan)
    for row in rows:
        for strategy in ('P0','P1','P2'):
            stream=tmp_path/'search'/row['task_id']/strategy;root=stream/'planning'/row['task_id']
            write(root/'selection.json',dict(task_id=row['task_id'],task_sha256=row['task_sha256'],selection_reads_final_actual=False,
                preferences={p:dict(selected_plan=None) for p in ('A','B')}))
            write(root/'candidate_registry.json',[]);write(root/'proposals.json',[]);write(stream/'planning_cost.json',dict(complete=True))
    with patch('v6_4.c4a_experiment.verify',return_value=plan),patch.object(mujoco,'mj_step',side_effect=AssertionError('no physics')):
        seal_dev(tmp_path)
    seal=verify_phase_selections(tmp_path/'validation')
    assert seal['phase']=='DEV' and seal['logical_actual_slots']==12 and len(seal['entries'])==6
    assert not (tmp_path/'validation/actual').exists()
    target=Path(seal['entries'][0]['selection_path']);target.write_text('{}')
    with pytest.raises(ValueError,match='changed'): verify_phase_selections(tmp_path/'validation')


def test_actual_alias_binds_original_initial_history(tmp_path):
    from v6_4.route_optimizer_protocol import parameter_plan
    from v6_4.closed_loop_warmstart_validation import cold_initial_history
    task=TaskSpec.from_dict(read(RELEASE/'snapshot/frozen_tasks/c1_mother_00_plus/task.json'))
    plan=parameter_plan(task,'v1',[0,0,0,0])
    for name in ('source_identity.json','frozen_execution_config.json','frozen_run_config.json'): write(tmp_path/name,dict(identity=name))
    a=actual_alias_identity(task,plan,tmp_path);h=cold_initial_history(task);h['policy']='different controller history'
    b=actual_alias_identity(task,plan,tmp_path,initial_history_identity=h)
    assert a!=b
