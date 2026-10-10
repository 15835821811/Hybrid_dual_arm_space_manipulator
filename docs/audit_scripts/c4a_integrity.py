"""Final frozen-byte, budget, source-isolation and actual-seal verification."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from v6_4.route_optimizer_protocol import read,write,sha,digest,TaskSpec
from v6_4.c4a_frozen import RELEASE
from v6_4.c4a_experiment import verify
from v6_4.route_candidate_evaluator import verify_seal
from v6_4.closed_loop_warmstart_validation import verify_phase_selections,verify_actual_slot


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    run=a.run.resolve();plan=verify(run)
    if not (run/'execution_complete.json').exists():raise RuntimeError('complete physical run required for final verification')
    expected=read(ROOT/'v6_4/c4a_evidence/audit_01/input_sha256.json')
    for relative,h in expected.items():
        if sha(RELEASE/relative)!=h:raise ValueError('C3 frozen artifact changed: '+relative)
    for name in ('frozen_execution_config.json','frozen_run_config.json'):
        if sha(run/name)!=sha(RELEASE/'snapshot'/name):raise ValueError('original control/execution config changed')
    phase=verify_phase_selections(run/'validation')
    if phase['phase']!='DEV' or phase['logical_actual_slots']!=12:raise ValueError('DEV phase/denominator differs')
    inspected=[];calls=slots=0;actual=[];paired=[]
    for taskrow in plan['tasks']:
        tid=taskrow['task_id'];all_rows={}
        task=TaskSpec.from_dict(read(run/taskrow['task_path']))
        for strategy in ('P0','P1','P2'):
            root=run/'search'/tid/strategy/'planning'/tid
            rows=read(root/'candidate_registry.json');selection=read(root/'selection.json');all_rows[strategy]=rows
            assert len(rows)<=8 and selection['budget']['shared_A_B_pool'] and digest(rows)==selection['registry_content_sha256']
            for r in rows:
                directory=root/'predictions'/r['candidate_id']
                if r['prediction_rollout_started']:
                    verify_seal(directory);start=read(directory/'started.json')
                    assert start['initial_state_sha256']==digest([task.initial_qpos,task.initial_qvel])
                    assert 'new runner/model/MjData/controller' in start['state_isolation']
                    calls+=1
                elif r['status']=='INITIALIZER_RAW_REJECTED':
                    assert not directory.exists() and r['prediction_steps']==0
                    assert not r['raw_seed_diagnostics']['raw_repaired'] and not r['raw_seed_diagnostics']['resampled']
                elif r.get('evidence_alias_of_candidate_id'):
                    original=next(x for x in rows if x['candidate_id']==r['evidence_alias_of_candidate_id'])
                    assert r['plan_sha256']==original['plan_sha256'] and not directory.exists() and not r['independent_physical_evidence']
                    assert r['costs']=={'cache_hit_zero_new_work':True}
            slots+=len(rows)
            inspected.append(dict(task_id=tid,strategy=strategy,slots=len(rows),registry_sha256=sha(root/'candidate_registry.json')))
            for preference in ('A','B'):
                slot=verify_actual_slot(run/'validation/actual'/tid/(strategy+'_'+preference))
                assert slot['phase']=='DEV' and slot['task_sha256']==task.sha256()
                actual.append(slot)
        assert [r['source'] for r in all_rows['P2'][:6]]==['initial']*4+['diffusion']*2
        for i in range(4):
            for k in ('plan_sha256','prediction_admissible','prediction_metrics'):
                if all_rows['P0'][i][k]!=all_rows['P2'][i][k]:raise ValueError('P2/P0 original rule evidence differs')
        for i,j in ((1,4),(3,5)):
            if all_rows['P1'][i]['raw_z_m']!=all_rows['P2'][j]['raw_z_m']:raise ValueError('P1/P2 learned proposals differ')
        paired.append(dict(task_id=tid,P0_P2_four_rule_seeds_and_physical_metrics_identical=True,P1_P2_raw_proposals_identical=True))
    assert slots<=48 and len(actual)==12
    value=dict(status='PASS',verified_utc=datetime.now(timezone.utc).isoformat(),
        C3_frozen_files_unchanged=len(expected),original_execution_configs_unchanged=True,
        frozen_current_sources_verified=True,streams=inspected,paired_checks=paired,
        candidate_slots=slots,independent_prediction_calls=calls,logical_actual_slots=len(actual),
        unique_actual_runs=sum(s['unique_run'] for s in actual),actual_aliases=sum(bool(s.get('alias_of_slot')) for s in actual),
        new_verification_physics_steps=0,new_model_forwards=0,new_training_updates=0)
    write(a.output,value);print(json.dumps(value))


if __name__=='__main__':main()
