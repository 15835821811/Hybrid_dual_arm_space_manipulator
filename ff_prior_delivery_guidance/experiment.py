"""One TRAIN comparison block. Unknown interrupted physics is never replayed."""
from pathlib import Path
import subprocess
import time
import zipfile
import numpy as np
from ff_prior.storage import read_json, file_digest, digest, RunLock
from ff_prior_pilot_train.storage import atomic_json, Ledger
from ff_prior.pipeline import audit, acceptance
from ff_prior.guidance import public_goal_initializer
from ff_prior.spline import Reference
from ff_prior_feasible.validation import validate
from ff_prior_generalization.experiment import execute
from ff_prior_origin_learn_bounded.learning import load
from ff_prior_latent.learning import Codec
from ff_prior_origin_learn_bounded.runtime import TaskOriginSceneCache
from .protocol import ROOT, OLD, BASE, PROTOCOL, METHODS
from .inputs import identity_for
from .sampling import candidate


def run(source, out, progress, resume=False, previous_elapsed=0.):
    from .reporting import report
    from .audit_results import verify
    out = Path(out).resolve()
    if out.parent != ROOT/'artifacts/v7_1' or out.name != 'delivery_guidance_smoke_001':
        raise ValueError('only frozen independent M1 output is allowed')
    started = time.time()
    deadline = started+max(0., PROTOCOL['wall_budget_s']-previous_elapsed)
    identity, (config, entries) = identity_for(source)
    with RunLock(out):
        if (out/'run.json').exists():
            if not resume or read_json(out/'run.json')['identity'] != identity:
                raise ValueError('frozen run identity changed')
        else:
            atomic_json(out/'run.json', {'identity': identity,
                'producer_head': subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                'TEST': False, 'training_updates': 0})
            atomic_json(out/'frozen_tasks.json', {'entries': entries, 'digest': digest(entries), 'before_any_comparison': True})
            with zipfile.ZipFile(out/'executed_source.zip', 'w', zipfile.ZIP_DEFLATED) as z:
                for p in identity['source_hashes']:
                    z.write(ROOT/p, p)
            progress('all_TRAIN_tasks_frozen', targets=8, candidate_slots=320, execution_units=64)
        tests = read_json(ROOT/'ff_prior_delivery_guidance/evidence/software_tests.json')
        if (tests['status'] != 'PASS' or tests['source_hashes'] != identity['source_hashes']
                or tests['junit_sha256'] != file_digest(ROOT/'ff_prior_delivery_guidance/evidence/tests.xml')):
            raise ValueError('software proof absent or stale')
        physical, canonical, initial, _ = audit(config)
        if physical['acceptance'] != read_json(BASE/'baseline_audit.json')['acceptance']:
            raise ValueError('inherited physical acceptance changed')
        if physical['mapping_left_inverse_error'] > 1e-12 or physical['initial_momentum_max'] > 1e-10:
            raise ValueError('physical contract audit failed')
        atomic_json(out/'physical_contract.json', physical)
        profile = acceptance(config)
        ledger = Ledger(out, {'identity_hash': digest(identity)})
        import torch
        torch.set_num_threads(2)
        bundle = load(OLD/'learning/goal_loss')
        mean_bundle = (bundle[0], Codec(read_json(OLD/'learning/weighted_mean.json')))
        cache = TaskOriginSceneCache(canonical, initial, profile, max_entries=8, max_scene_models=4)

        def budget():
            if time.time() >= deadline-120:
                raise RuntimeError('TOTAL_WALL_BUDGET_EXHAUSTED')
            if sum(p.stat().st_size for p in out.rglob('*') if p.is_file()) >= PROTOCOL['max_output_bytes']:
                raise RuntimeError('OUTPUT_STORAGE_BUDGET_EXHAUSTED')

        # Representation diagnostics precede, and cannot choose, comparison tasks.
        for e in entries:
            rr, ii, cc, transfer = cache.get(e)
            task, sid = e['task'], e['sample_id']
            base = public_goal_initializer(task, rr, ii)
            atomic_json(out/'transfer_audits'/f'{sid}.json', dict(transfer, public_base_hash=digest(base)))
            with np.load(ROOT/e['arrays']) as a:
                teacher = a['Z'].copy()
            if Reference(teacher, task['condition']['initial_features'], task['condition']['duration']).reference_hash != e['reference_hash']:
                raise ValueError('teacher reference identity changed')
            roundtrip = bundle[1].decode(bundle[1].encode(teacher, base), base)
            for name, z in [('teacher', teacher), ('roundtrip', roundtrip)]:
                key = f'diagnostic_{sid}_{name}'
                path = out/'diagnostics'/sid/f'{name}.json'
                if key in ledger.data['completed']:
                    continue
                budget()
                ledger.begin(key, 'diagnostic', 600., 16)
                progress('codec_world_goal_and_geometry', task=sid, representation=name)
                before = rr.calls.copy()
                try:
                    r = dict(validate(z, task, rr, ii, cc, profile, min(deadline-120, time.time()+30.)),
                             status='DONE', Z=z, task_hash=task['task_hash'], compression_max_rad=float(abs(teacher-roundtrip).max()))
                except Exception as error:
                    import traceback
                    r = {'status': 'ERROR', 'eligible': False, 'reason': repr(error), 'traceback': traceback.format_exc()}
                r['model_calls'] = {k: rr.calls[k]-before[k] for k in before}
                atomic_json(path, r)
                ledger.finish({'status': r['status']}, r['model_calls'])
        diagnostic = []
        for e in entries:
            pair = {}
            for name in ('teacher', 'roundtrip'):
                p = out/'diagnostics'/e['sample_id']/f'{name}.json'
                pair[name] = read_json(p) if p.exists() else {'status': 'INTERRUPTED', 'eligible': False}
            diagnostic.append(pair)
        original_ok = all(p['teacher']['status']=='DONE' and p['teacher']['eligible'] for p in diagnostic)
        representation_ok = original_ok and all(p['roundtrip']['status']=='DONE' and p['roundtrip']['eligible'] for p in diagnostic)
        atomic_json(out/'representation_gate.json', {'original_teachers_qualified': original_ok,
            'latent_roundtrip_qualified': representation_ok, 'late4_enabled': representation_ok,
            'teacher_eligible': sum(p['teacher']['eligible'] for p in diagnostic),
            'roundtrip_eligible': sum(p['roundtrip']['eligible'] for p in diagnostic), 'denominator': 8,
            'basis_retrained': False, 'fallback_on_representation_loss': True})
        if not original_ok:
            atomic_json(out/'shared_block.json', {'reason': 'ORIGINAL_TEACHER_RECHECK_FAILED', 'dependent_comparison': 'BLOCKED'})
        else:
            for sj, seed in enumerate(PROTOCOL['sampling_seeds']):
                for ti, e in enumerate(entries):
                    sid, task = e['sample_id'], e['task']
                    rr, ii, cc, _ = cache.get(e)
                    base = public_goal_initializer(task, rr, ii)
                    for method in METHODS[ti%4:]+METHODS[:ti%4]:
                        selected_bundle = None if method=='structured' else mean_bundle if method=='learned_mean' else bundle
                        records = []
                        for k in range(5):
                            key = f'candidate_{sj}_{sid}_{method}_{k}'
                            path = out/'candidates'/str(sj)/sid/method/f'{k:02d}.json'
                            if key not in ledger.data['completed']:
                                budget()
                                ledger.begin(key, 'candidate', 7200., 320)
                                progress('TRAIN_paired_candidates', seed=sj+1, task=ti+1, method=method, candidate=k+1)
                                seed_k = seed+ti*101+k*997
                                if method=='late4_diffusion' and not representation_ok:
                                    r = {'status': 'BLOCKED_REPRESENTATION', 'eligible': False, 'method': method,
                                         'candidate_id': k, 'seed': seed_k, 'task_hash': task['task_hash'],
                                         'reason': 'frozen pre-comparison codec gate failed', 'model_calls': {}, 'actual_network_calls': 0}
                                else:
                                    r = candidate(method, k, task, base, seed_k, selected_bundle, rr, ii, cc, profile, deadline-120)
                                r.update(sample_id=sid, split='TRAIN', seed_group=sj)
                                atomic_json(path, r)
                                ledger.finish({'status': r['status']}, r['model_calls'])
                            if path.exists() and ledger.data['completed'][key]['status']!='INTERRUPTED':
                                records.append(read_json(path))
                        key = f'execution_{sj}_{sid}_{method}'
                        directory = out/'executions'/str(sj)/sid/method
                        if key in ledger.data['completed']:
                            continue
                        budget()
                        chosen = min((r for r in records if r['eligible']), key=lambda r:(r['score'],r['candidate_id']), default=None)
                        ledger.begin(key, 'execution', 7200., 64)
                        atomic_json(directory/'selection.json', {'selected_id': chosen['candidate_id'] if chosen else None,
                            'task_hash': task['task_hash'], 'selection_before_execution': True, 'rule': PROTOCOL['selection']})
                        before = rr.calls.copy()
                        progress('selected_native_torque_and_replay', seed=sj+1, task=ti+1, method=method)
                        if chosen:
                            r = execute(directory, task, chosen['Z'], rr, ii, cc, profile, deadline-120)
                        else:
                            r = {'status': 'NO_CANDIDATE'}
                            atomic_json(directory/'outcome.json', r)
                        ledger.finish(r, {k: rr.calls[k]-before[k] for k in before})
                    report(out, 'RUNNING')
        atomic_json(out/'runtime_cache.json', cache.stats())
        progress('read_only_raw_audit')
        verification = verify(out, identity)
        atomic_json(out/'independent_verification.json', verification)
        return report(out, 'FINISHED' if verification['status']=='PASS' else 'FINISHED_WITH_BLOCKED_ITEMS')
