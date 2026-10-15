"""Verify frozen-comparison invariants and summarize every candidate, including failures."""
from collections import Counter
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET
import numpy as np
from ff_prior.storage import read_json,atomic_json,digest,file_digest
from ff_prior.contracts import EXECUTION_GATES,assert_frozen
from ff_prior.pipeline import source_manifest


def main():
    base=Path('artifacts/v7_1')
    source=base/'smoke_002'
    out=base/'diffusion_repair'
    methods=read_json(source/'benchmark.json')['methods']
    tasks=read_json(source/'benchmark_tasks.json')
    result={'scope':'adaptive DEV debugging; not a fresh held-out benefit test',
        'reference_repository_commit':'900d781ba63f0123dff95e49a76d51cb49a74803',
        'frozen_tasks_hash':digest(tasks),'runs':{},'pilot':'NOT_RUN','TEST':'NOT_RUN','benefit':'NOT_ESTABLISHED'}
    for name in ('smoke_002','smoke_diffusion_repair_001','smoke_diffusion_repair_002'):
        p=base/name
        assert read_json(p/'benchmark_tasks.json')==tasks
        data={}
        for method in methods:
            failures=Counter(); records=[]; networks=0; cost_evals=0
            for i,task in enumerate(tasks):
                plan=read_json(p/'benchmark'/f'benchmark_{i:03d}_{method}'/'planning.json')
                assert plan['task_hash']==task['task']['task_hash']
                assert plan['public_conditions_hash']==digest(task['task']['condition'])
                assert plan['selection_before_execution'] and len(plan['candidates'])==5
                assert not plan['private_teacher_fields_accessed']
                cost_evals+=plan['full_path_cost_evaluations']
                eligible=[]
                for c in plan['candidates']:
                    assert 'error' not in c
                    assert c['seed']==20261017+i*101+c['candidate_id']*997
                    assert c['sampling']['actual_network_calls']==(0 if method.startswith('structured') else 16)
                    networks+=c['sampling']['actual_network_calls']
                    guides=c['sampling']['guidance']
                    assert len(guides)==(0 if method=='diffusion_no_guidance' else 2)
                    for g in guides:
                        assert g['directions']==8 and g['centered_evaluations']==16
                        assert g['initial_cost_terms']['coupled_entire_path']
                    if method=='diffusion_in_denoising_guidance':
                        assert [g['iteration'] for g in guides]==[14,15]
                    if name!='smoke_002' and not method.startswith('structured'):
                        assert max(s['decomposition_residual_after'] for s in c['sampling']['steps'])<1e-12
                    if name=='smoke_diffusion_repair_002' and not method.startswith('structured'):
                        assert len(c['sampling']['clean_prior_projection'])==16
                        assert c['sampling']['clean_prior_fit_split']=='TRAIN'
                    if c['eligible']: eligible.append(c)
                    v=c['validation']
                    failures.update(k for k,ok in v['gates'].items() if not ok)
                    records.append({'task':i,'candidate':c['candidate_id'],'model_valid':v['validation_level']=='MODEL_VALID',
                        'goal_valid':c['frozen_goal_pass'],'eligible':c['eligible'],
                        'max_goal_position_error_m':max(v.get('frozen_goal_position_error_m',[]),default=None),
                        'max_goal_rotation_error_rad':max(v.get('frozen_goal_rotation_error_rad',[]),default=None),
                        'geometry_step_m':v.get('geometry',{}).get('maximum_geometry_step_m'),
                        'minimum_clearance_m':v.get('geometry',{}).get('minimum_clearance_m')})
                expected=min(eligible,key=lambda x:(x['score'],x['candidate_id']))['candidate_id'] if eligible else None
                assert expected==plan['selected_id']
            data[method]={'candidates':len(records),'model_valid':sum(r['model_valid'] for r in records),
                'frozen_goal_valid':sum(r['goal_valid'] for r in records),'failed_gates':dict(failures),
                'network_calls':networks,'full_path_cost_evaluations':cost_evals,'records':records}
        reports=[read_json(f) for f in (p/'benchmark').glob('*/execution/report.json')]
        assert len(reports)==1
        for report in reports:
            assert all(report['gates'][g] is True for g in EXECUTION_GATES)
            assert report['replay']['passed'] and max(report['replay']['maximum_errors'].values())==0
        result['runs'][name]={'methods':data,'execution_count':len(reports),'replay_pass':len(reports),
            'benchmark_summary':read_json(p/'benchmark.json'),'measured_summary':read_json(p/'summary.json')}
    final=base/'smoke_diffusion_repair_002'
    prior=read_json(final/'learning/clean_prior.json')
    dataset=read_json(source/'dataset.json')
    assert prior['metadata']['sample_ids']==dataset['eligible']['TRAIN']
    assert prior['metadata']['rank']==5
    m=read_json(final/'run.json')
    assert source_manifest()==m['source_hashes']
    assert all(file_digest(source/rel)==h for rel,h in m['identity']['source_files'].items())
    before=read_json(out/'resume_before.json'); after=read_json(final/'ledger.json')
    assert all(before[k]==after[k] for k in ('attempts','calls','completed'))
    result['resume_no_duplicate_work']=True
    result['physical_profiles_identical']=all(read_json(base/n/'run.json')['physical_acceptance']==read_json(source/'baseline_audit.json')['acceptance'] for n in ('smoke_diffusion_repair_001','smoke_diffusion_repair_002'))
    assert result['physical_profiles_identical']
    suites=ET.parse(out/'tests_final.xml').getroot().findall('testsuite')
    assert sum(int(s.attrib['tests']) for s in suites)==48
    assert all(int(s.attrib[k])==0 for s in suites for k in ('failures','errors','skipped'))
    result['tests_passed']=48
    result['temporal_prior']=prior['metadata']
    old_patch=Path('artifacts/v7_1/baseline/user_changes.patch').read_text(encoding='utf-8-sig').replace('\r\n','\n').strip()
    current=subprocess.check_output(['git','diff','--binary','--','.gitattributes','.gitignore','v6_lite/hierarchical_qp.py','v6_lite/run_v6_lite.py'],text=True,encoding='utf-8').replace('\r\n','\n').strip()
    assert current==old_patch
    result['user_tracked_changes_preserved']=True
    result['source_run_inputs_preserved']=True
    atomic_json(out/'comparison_audit.json',result)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,4.2))
    for name,label in (('diagnosis_before.json','Original epsilon'),('diagnosis_final.json','Repaired v')):
        bands=read_json(out/name)['bands']['VAL']
        axes[0].semilogy([r['t'] for r in bands],[r['x0_rmse_rad'] for r in bands],'-o',label=label)
    axes[0].set(xlabel='Noise timestep',ylabel='Clean control RMSE (rad)',title='VAL denoising diagnosis'); axes[0].legend()
    names=list(result['runs']); x=np.arange(3); width=.34
    for shift,method,label in ((-width/2,'diffusion_in_denoising_guidance','Guided'),(width/2,'diffusion_no_guidance','Unguided')):
        axes[1].bar(x+shift,[result['runs'][n]['methods'][method]['model_valid'] for n in names],width,label=label)
    axes[1].set_xticks(x,['Original','Numerics','+Temporal']); axes[1].set(ylim=(0,20),ylabel='MODEL_VALID / 20',title='Same frozen DEV tasks'); axes[1].legend()
    for method,label in (('diffusion_in_denoising_guidance','Guided'),('diffusion_no_guidance','Unguided')):
        values=[r['max_goal_position_error_m']*1000 for r in result['runs'][names[-1]]['methods'][method]['records']]
        axes[2].plot(range(20),values,'o-',label=label)
    axes[2].axhline(5,color='red',ls='--',label='5 mm frozen limit')
    axes[2].set(xlabel='Candidate (4 tasks x 5)',ylabel='Worse-arm position error (mm)',title='Remaining goal failures'); axes[2].legend()
    for ax in axes: ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(out/'diagnosis.png',dpi=160); plt.close(fig)
    for name,entry in result['runs'].items():
        print(name,{m:{k:v[k] for k in ('model_valid','frozen_goal_valid','failed_gates')} for m,v in entry['methods'].items()})
    return result


if __name__=='__main__': main()
