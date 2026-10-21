"""Read-only reconstruction of saved identities, updates and qualification."""
import zipfile,hashlib
import numpy as np
from ff_prior.storage import read_json,digest
from ff_prior_pilot_train.storage import atomic_json
from ff_prior_fixed.correction import pose_vector
from ff_prior.coupled_predictor import pose_errors
from ff_prior_delivery_guidance.audit_results import check_candidate_validation
from ff_prior_complex.tools.audit_completed import audit_physics
from ff_prior_latent.learning import Codec
from ff_prior_capture_feasibility.solver import basis
from .protocol import P,SOURCE
from .inputs import identity
from .solver import accept,metric

def verify(out,ident):
    errors=[];columns=updates=validations=executions=physics=0
    try:
        assert identity()==ident
        task=ident['selection']['entry']['task'];assert read_json(out/'frozen_task.json')==task
        with zipfile.ZipFile(out/'executed_source.zip') as archive:
            assert all(hashlib.sha256(archive.read(p)).hexdigest()==h for p,h in ident['source_hashes'].items())
        assert read_json(out/'sanity_uniform.json')['status']=='PASS'
        with np.load(out/'sanity_uniform.npz') as a:
            assert np.max(abs(a['old_qpos']-a['new_qpos']))<1e-12 and np.max(abs(a['old_ee']-a['new_ee']))<1e-12
        codec=Codec(read_json(SOURCE/'learning/aligned/codec.json'));limits=read_json(out/'physical_limits.json')
        for seed in ident['selection']['seeds']:
            assert np.array_equal(read_json(out/'teachers'/seed['id']/'seed.json')['actual_start_Z'],seed['Z'])
        for p in sorted(out.glob('**/iterations/*/record.json')):
            r=read_json(p)
            with np.load(p.with_name('linearization.npz')) as a:
                z=a['Z'];D=basis(r['direction_basis'],codec);M=D.reshape(len(D),-1).T
                e=pose_vector(a['current_ee'],task['condition']['goals'])
                assert np.allclose(e,a['goal'],atol=1e-12,rtol=0)
                for j,h in enumerate(a['steps']):
                    if h:
                        assert np.allclose((pose_vector(a['perturb_ee'][j],task['condition']['goals'])-e)/h,a['jg'][:,j],atol=1e-10,rtol=1e-10)
                        assert np.allclose((a['perturb_distances'][j]-a['active_distances'])/h,a['jd'][:,j],atol=1e-10,rtol=1e-10)
                    else:assert not np.any(a['jg'][:,j]) and not np.any(a['jd'][:,j])
                assert all(t in a['phase'] for t,_,_ in r['active_pairs'])
                assert all(c['actual_intervals']<=c['base_intervals']+P['active_pair_cap'] for c in r['sparse_agreement'])
                c=r['sparse_agreement'][-1]
                assert max(c['position_m'])<=.001 and max(c['rotation_rad'])<=P['sparse_agreement_rad'] and c['distance_difference_m']<=.001
                sol=r['solver'];step=a['step'];delta=np.einsum('i,ijk->jk',step,D)
                assert sol['initialization']=='ZERO_FEASIBLE' and sol['returned_feasible']
                assert np.allclose(step,P['trust_rad']*np.asarray(sol['returned_u']),atol=1e-14,rtol=0)
                assert max(abs(delta).ravel())<=.15+1e-8
                assert np.all(z+delta>=np.asarray(limits['lower'])-1e-8) and np.all(z+delta<=np.asarray(limits['upper'])+1e-8)
                assert np.max(np.sum((e+a['jg']@step).reshape(4,3)**2,axis=1)/np.asarray(sol['pose_limits'])**2)<=1+1e-8
                assert digest(z)==r['before_Z_hash'] and digest(a['next_Z'])==r['after_Z_hash']
                accepted=[t for t in r['trials'] if t.get('accepted')]
                assert bool(accepted)==r['accepted']
                for j,t in enumerate(r['trials'],1):
                    if t['status']!='EVALUATED':continue
                    with np.load(p.parent/f'trial_{j:02d}.npz') as v:
                        assert np.array_equal(v['Z'],z+t['scale']*delta)
                        trial={'goal':pose_vector(v['ee'],task['condition']['goals']),'minima':v['minima']}
                        old={'goal':pose_vector(a['dense_ee'],task['condition']['goals']),'minima':a['dense_minima']}
                        assert bool(accept(old,trial))==t['accepted']
                        assert np.isclose(metric(trial)['minimum_clearance_m'],t['minimum_clearance_m'],atol=1e-12)
                if accepted:
                    assert np.array_equal(a['next_Z'],z+accepted[0]['scale']*delta);updates+=1
                else:assert np.array_equal(a['next_Z'],z)
                columns+=len(a['steps'])
        records={}
        for p in out.glob('**/validation.json'):
            r=read_json(p);check_candidate_validation(r,task);assert r['task_hash']==task['task_hash'] and not r['goal_relabel']
            if r['validation_grids']:
                with np.load(p.with_name('prediction.npz')) as pred:
                    pos,rot=pose_errors(pred['ee'][-1],task['condition']['goals'])
                    assert np.allclose(pos,r['validation']['frozen_goal_position_error_m'],atol=1e-12,rtol=0)
                    assert np.allclose(rot,r['validation']['frozen_goal_rotation_error_rad'],atol=1e-12,rtol=0)
                if r['eligible']:assert r['validation']['geometry']['maximum_geometry_step_m']<=.002
            records[r['reference_hash']]=r;validations+=1
        choices=[(s['id'],read_json(out/'teachers'/s['id']/'validation.json')) for s in ident['selection']['seeds']]
        selected=read_json(out/'teacher_execution_selection.json')
        expected=[sid for sid,r in sorted((v for v in choices if v[1]['eligible']),key=lambda v:(v[1]['score'],v[0]))[:2]]
        assert selected['ids']==expected and selected['candidate_digest']==digest(choices)
        for p in out.glob('**/outcome.json'):
            selection=read_json(p.with_name('selection.json'));result=read_json(p)
            assert selection['task_hash']==task['task_hash'] and selection['before_execution']
            if result['status']=='NO_CANDIDATE':assert not selection['eligible'];continue
            assert records[selection['reference_hash']]['eligible']
            r,steps=audit_physics(p.parent,task,selection['reference_hash'])
            assert r['level']==result['status'];executions+=1;physics+=steps
        ledger=read_json(out/'ledger.json');budget=read_json(out/'budget.json')
        assert ledger['inflight'] is None and not ledger.get('unresolved_call_accounting')
        assert sum(v for k,v in ledger['calls'].items() if k.endswith('.coupled'))==budget['actual_coupled']<=P['coupled_cap']
        assert sum(v for k,v in ledger['calls'].items() if k.endswith('.physics_steps'))==physics
        assert budget['integration_intervals']<=P['integration_cap'] and executions<=5
    except Exception as e:
        import traceback
        errors.append({'error':repr(e),'traceback':traceback.format_exc()})
    return {'status':'PASS' if not errors else 'FAIL','errors':errors,'difference_columns':columns,
      'accepted_updates':updates,'validation_records':validations,'actual_execution_records':executions,
      'native_plus_replay_steps':physics,'scope':'saved evidence algebra, not new simulation'}

def report(out,audit):
    rows={s['id']:read_json(out/'teachers'/s['id']/'validation.json') for s in read_json(out/'run.json')['identity']['selection']['seeds']}
    executions={str(p.parent.relative_to(out)):read_json(p)['status'] for p in out.glob('**/outcome.json')}
    summary={'status':'FINISHED' if audit['status']=='PASS' else 'FINISHED_WITH_BLOCKED_ITEMS','raw_audit':audit,
      'teacher_slots':2,'model_qualified':sum(r['eligible'] for r in rows.values()),'executions':executions,
      'teachers':{k:{'eligible':v['eligible'],'validation':v['validation']} for k,v in rows.items()},
      'budget':read_json(out/'budget.json'),'TRAIN_added':0,'TEST':False,'physics_relaxed':False,
      'infeasibility_proven':False,'independent_claim_review':'PENDING'}
    atomic_json(out/'summary.json',summary)
    text=['# 00/07精细碰撞反馈：后台结果',f"状态{summary['status']}；原始审计{audit['status']}，独立审查待收尾。",
      f"完整模型合格{summary['model_qualified']}/2；执行结果{executions}。",
      '保持原目标、物理门及旧结果；无训练、扩库或TEST。有限失败不证明不可达。',
      '最终分析将写入ff_prior_capture_refine/docs/ANALYSIS.md。']
    (out/'REPORT.md').write_text('\n\n'.join(text)+'\n',encoding='utf-8')
    return summary
