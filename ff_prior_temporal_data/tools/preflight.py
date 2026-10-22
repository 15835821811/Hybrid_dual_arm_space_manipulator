"""Read-only selection, rejection and real saved representation checks; zero mj_step."""
import copy
import json
import time
import tempfile
from pathlib import Path
import numpy as np
from scipy.interpolate import BSpline
from ff_prior.storage import read_json
from ff_prior_pilot_train.storage import atomic_json
from ff_prior_temporal_data.protocol import ROOT,PACKAGE,P,SOURCE
from ff_prior_temporal_data.inputs import identity,selection
from ff_prior_temporal_data.experiment import Budget,training_gate
from ff_prior_dynamic_margin.protocol import P as FIT

def main():
    checks=[]
    def check(name,value):
        assert value,name;checks.append(name)
    ident=identity();inventory=read_json(SOURCE)['selected'];rows=selection(inventory)
    check('fixed_selection_8_train_4_val',len(rows)==12 and sum(r['split']=='train' for r in rows)==8)
    check('exact_task_seeds',[r['source']['seed'] for r in rows]==[
        31000000,31104729,31209458,31314187,31418916,31523645,31628374,31733103,
        32570935,32675664,99378320,99483049])
    check('exactly_four_reuses_eight_new',sum(r['reuse'] is not None for r in rows)==4)
    wrong=copy.deepcopy(inventory);wrong['validation'][0]['split']='train'
    try:selection(wrong)
    except AssertionError:checks.append('wrong_split_rejected')
    else:raise AssertionError('wrong_split_accepted')
    statuses=[dict(split=r['split'],status='EXECUTION_VERIFIED') for r in rows]
    check('gate_requires_8_4',training_gate(statuses)['passed'])
    bad=copy.deepcopy(statuses);bad[-1]['status']='MODEL_VALID'
    check('model_only_does_not_satisfy_val',not training_gate(bad)['passed'])
    bad=copy.deepcopy(statuses);bad[0]['status']='EXECUTION_FAILED'
    check('failed_train_not_accepted',not training_gate(bad)['passed'])
    for key in ['duration','dt','steps_each','acceleration_fraction']:
        check('frozen_reference_contract_'+key,P[key]==FIT[key])
    for row in rows[:4]:
        previous=ROOT/row['reuse']
        with np.load(previous/'reference.npz',allow_pickle=False) as a:
            spline=BSpline(a['knots'],a['coefficients'],3,extrapolate=False)
            check(row['id']+'_full_representation',a['coefficients'].shape==(273,17) and len(a['knots'])==277)
            check(row['id']+'_contractual_endpoint_finite',np.isfinite(spline(27.)).all())
            check(row['id']+'_initial_q_dq_preserved',np.max(abs(spline(0)-a['source_q'][0]))<1e-10 and np.max(abs(spline(0,1)-a['source_dq'][0]))<1e-10)
    with tempfile.TemporaryDirectory() as temp:
        budget=Budget(Path(temp));budget.native=P['max_native_replay_steps']
        try:budget.step('native')
        except TimeoutError:checks.append('step_budget_before_increment')
        else:raise AssertionError('step_budget_not_enforced')
        check('budget_counter_not_overrun',budget.native==P['max_native_replay_steps'])
        budget.native=0;budget.deadline=time.monotonic()-1
        try:budget.step('native')
        except TimeoutError:checks.append('wall_budget_before_step')
        else:raise AssertionError('wall_budget_not_enforced')
    check('planned_steps_within_limit',8*2*P['steps_each']==P['max_native_replay_steps'])
    # Raw full-state pairs dominate output; other traces + zip remain below 5 GiB.
    raw_bytes=8*2*(13500+1)*(836+81+79+67)*8
    check('raw_size_estimate_has_margin',raw_bytes*1.5<P['max_bytes'])
    check('identity_unchanged_after_readonly_tests',identity()==ident)
    result=dict(passed=True,count=len(checks),checks=checks,source_hashes=ident['source_hashes'],
        input_files=len(ident['binding']['files']),actual_physics_steps=0,training_updates=0,
        raw_trace_size_estimate_bytes=raw_bytes,not_a_physics_pass=True)
    atomic_json(PACKAGE/'evidence/software_tests.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='source_hashes'},ensure_ascii=False))

if __name__=='__main__':main()
