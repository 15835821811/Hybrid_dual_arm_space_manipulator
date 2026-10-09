"""Post-freeze maintenance: missing identifiers cannot alias rejected seeds."""
import copy
import pytest
from v6_4.search_effect_teacher import summarize_teacher_pair


def row(cid, slot, **identity):
    return dict(candidate_id=cid, initializer_slot=slot, prediction_admissible=True,
        prediction_task_passed=True, online_guards_passed=True, prediction_steps=13500,
        prediction_metrics=dict(I_support=.1,L_full=1.,d_support=.031,
            clearance_status='MEASURED',native_state_count=13501,saved_horizon_s=27.),
        proposal_lineage=[],**identity)


def inputs():
    z=[[0.,0.] for _ in range(6)]; z[2][0]=.001
    pair=dict(seed_pair_id='pair',task_id='task',task_sha256='task-sha',mother_id='mother',combination='T_local',
        proposals={str(slot):dict(raw_z_m=copy.deepcopy(z),family=family,preference=pref,
            teacher_construction=dict(construction_status='LOCAL_QUALIFIED_ROUTE_TEACHER'))
            for slot,pref,family in [(1,'A','v1'),(3,'B','v2')]})
    selection=dict(budget=dict(candidate_budget=8,slots_consumed=4,stop_reason='CANDIDATE_BUDGET_EXHAUSTED'),
        preferences=dict(A=dict(source_candidate_id='C00'),B=dict(source_candidate_id='C03')))
    rows=[row('C00',0,plan_sha256='rule0',search_content_key='rule0-key'),
          dict(candidate_id='C01',initializer_slot=1,status='RAW_REJECTED',plan_sha256=None,
               search_content_key=None,prediction_admissible=False),
          row('C02',2,plan_sha256='rule2',search_content_key='rule2-key'),
          row('C03',3,plan_sha256='seed3',search_content_key='seed3-key')]
    return pair,selection,rows


@pytest.mark.parametrize('identity',[{'content_key':'seed3-key'},{'plan_sha256':'seed3'}])
def test_missing_other_identity_does_not_bind_earlier_rejection(identity):
    pair,selection,rows=inputs()
    proposal=dict(initializer_slot=3,raw_seed_diagnostics=dict(raw_legal=True),**identity)
    result=summarize_teacher_pair(pair,selection,rows,{'A':None,'B':None},[proposal])
    assert result['endpoints']['A']['label_status']=='RULE_ONLY'
    assert result['endpoints']['B']['direct_seed_qualified'] is True
    labels=result['initializer_effect_labels']
    assert len(labels)==1 and labels[0]['preference']=='B'
    assert labels[0]['source_refs'][0]['seed_candidate_id']=='C03'
    assert labels[0]['z_m']==pair['proposals']['3']['raw_z_m']


@pytest.mark.parametrize('mode',['rule_only','no_plan','raw_rejected','missing_both_ids'])
def test_fallback_never_creates_unearned_effect_labels(mode):
    pair,selection,rows=inputs()
    proposal=dict(initializer_slot=3,raw_seed_diagnostics=dict(raw_legal=True),content_key='seed3-key')
    if mode=='rule_only':
        proposal['content_key']='rule2-key'; selection['preferences']['B']['source_candidate_id']='C02'
    elif mode=='no_plan':
        selection['preferences']['B']['source_candidate_id']=None
    elif mode=='raw_rejected':
        proposal['raw_seed_diagnostics']['raw_legal']=False
    else:
        proposal.pop('content_key')
    result=summarize_teacher_pair(pair,selection,rows,{'A':None,'B':None},[proposal])
    assert result['initializer_effect_labels']==[]
    assert result['endpoints']['B']['positive_supervision'] is False
