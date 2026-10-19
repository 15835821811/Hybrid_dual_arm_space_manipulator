"""One metadata-only selection, with fixed goals and preselected execution units."""
from collections import Counter
import hashlib
import json
import numpy as np
from ff_prior.storage import read_json, file_digest, digest
from ff_prior_pilot_train.storage import atomic_json
from ff_prior.contracts import assert_frozen
from ff_prior.spline import from_arrays
from ff_prior_loop.data import assert_disjoint
from .protocol import ROOT, SOURCE, M1, SELECTION, POPULATION, PROTOCOL


def choose(pool, retained, origin_counts):
    """Cover rare initial states and world positions; no candidate outcomes."""
    chosen = list(retained)
    for obstacle in (False, True):
        for _ in range(12):
            used_tasks = {r['task']['task_hash'] for r in chosen}
            used_families = {r['family_id'] for r in chosen}
            candidates = [r for r in pool if bool(r['scene']['objects'])==obstacle
                          and r['task']['task_hash'] not in used_tasks and r['family_id'] not in used_families]
            used_origins = {r['origin_hash'] for r in chosen}
            novel = [r for r in candidates if r['origin_hash'] not in used_origins]
            if novel:
                candidates = novel
            if not candidates:
                break
            features = np.asarray([r['coverage_features'] for r in chosen])
            def rank(r):
                distance = float(np.min(np.mean((features-r['coverage_features'])**2,axis=1)))
                return (-distance/np.sqrt(origin_counts[r['origin_hash']]), digest({'M2_target':r['parent_sample_id']}))
            chosen.append(min(candidates,key=rank))
    return chosen


def freeze():
    if SELECTION.exists() or POPULATION.exists():
        raise ValueError('selection already frozen; never reselect after outcomes')
    index = read_json(SOURCE/'training_index.json')['records']
    assert_disjoint(index)
    if len(index)!=13886 or sum(e['split']=='TRAIN' for e in index)!=13838:
        raise ValueError('unexpected inherited population')
    m1 = read_json(M1/'summary.json')
    if not m1['selection']['M2_ready'] or m1['audit']!='PASS':
        raise ValueError('M1 does not permit M2')
    retained_ids = [e['sample_id'] for e in read_json(M1/'frozen_tasks.json')['entries']]
    origin_counts = Counter()
    pool, references = [], set()
    metadata_sha = hashlib.sha256()
    fallback_arrays = 0
    for e in index:
        if e['split']!='TRAIN':
            continue
        raw = (ROOT/e['record']).read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        if sha!=e['record_sha256']:
            raise ValueError('TRAIN metadata source changed: '+e['sample_id'])
        metadata_sha.update(e['sample_id'].encode()+sha.encode())
        row = json.loads(raw.decode('utf-8-sig'))
        ref_hash = row.get('reference_hash')
        if ref_hash is None:
            if file_digest(ROOT/e['arrays'])!=e['arrays_sha256']:
                raise ValueError('legacy reference arrays changed')
            with np.load(ROOT/e['arrays']) as a:
                ref_hash = from_arrays(a).reference_hash
            fallback_arrays += 1
        references.add(ref_hash)
        condition = row.get('condition', row.get('task',{}).get('condition'))
        if condition is None:
            continue
        q0 = np.asarray(condition['initial_features'])
        origin = digest(q0)
        origin_counts[origin] += 1
        if 'task' not in row or e['execution_status']=='FAIL' or not row.get('training_eligible',True):
            continue
        task = row['task']
        assert_frozen(task)
        if condition != task['condition']:
            raise ValueError('public condition differs from task')
        world = np.asarray(condition['goals'])[:,:3,3].reshape(-1)
        pool.append({'parent_sample_id':e['sample_id'], 'family_id':e['family_id'], 'split':'TRAIN',
                     'task':task, 'scene':row.get('scene',{'objects':[]}), 'origin_hash':origin,
                     'coverage_features':np.r_[q0/.1,world/.1].tolist(), 'source_entry':e})
    lookup = {r['parent_sample_id']:r for r in pool}
    retained = [lookup[sid] for sid in retained_ids]
    chosen = choose(pool,retained,origin_counts)
    entries = []
    for i,r in enumerate(chosen):
        e = dict(r, target_id=f'delivery_target_{i:02d}', target_slot=i,
                 source_origin_rows=origin_counts[r['origin_hash']], frozen_before_generation=True,
                 goal_relabelled=False, inference_uses_teacher=False)
        if file_digest(ROOT/e['source_entry']['arrays'])!=e['source_entry']['arrays_sha256']:
            raise ValueError('selected source arrays changed')
        entries.append(e)
    selected = []
    for obstacle in (False, True):
        group = [e for e in entries if bool(e['scene']['objects'])==obstacle]
        selected += [e['target_id'] for e in sorted(group,key=lambda e:digest({'M2_execution':e['target_id']}))[:8]]
    POPULATION.parent.mkdir(parents=True,exist_ok=False)
    atomic_json(POPULATION, {'existing_reference_hashes':sorted(references), 'TRAIN_metadata_digest':metadata_sha.hexdigest(),
        'metadata_rows_checked':13838, 'legacy_arrays_read_for_missing_reference_hash':fallback_arrays,
        'origin_counts':dict(origin_counts), 'TEST_read':False, 'selection_used_method_results':False})
    proof = {'protocol':PROTOCOL, 'entries':entries, 'entries_digest':digest(entries), 'execution_target_ids':selected,
        'selection_before_generation':True, 'source_index_sha256':file_digest(SOURCE/'training_index.json'),
        'population_path':POPULATION.relative_to(ROOT).as_posix(), 'population_sha256':file_digest(POPULATION),
        'counts':{'targets':len(entries),'obstacle':sum(bool(e['scene']['objects']) for e in entries),
                  'retained':len(retained),'novel_initial_states':len({e['origin_hash'] for e in entries}),
                  'pool':len(pool),'execution_units':len(selected)}, 'sampler':m1['selection']['method'],
        'scope':'TRAIN only; existing fixed tasks, no new world goals or heldout arrays'}
    atomic_json(SELECTION,proof)
    print(json.dumps(proof['counts']))


if __name__=='__main__':
    freeze()
