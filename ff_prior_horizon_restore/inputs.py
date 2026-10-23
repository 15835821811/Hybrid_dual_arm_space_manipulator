"""One deterministic saved branch, exact identities; no historical resimulation."""
import ast
import sys
import numpy as np
from ff_prior.storage import read_json,file_digest,digest
from ff_prior.contracts import assert_frozen
from ff_prior_pilot_train.storage import atomic_json,atomic_npz
from .protocol import ROOT,PACKAGE,OLD,P


def sources():
    pending=list(PACKAGE.glob('*.py'));found=set()
    def add(module):
        parts=module.split('.')
        for i in range(1,len(parts)+1):
            p=ROOT.joinpath(*parts[:i])
            for path in (p.with_suffix('.py'),p/'__init__.py'):
                if path.is_file() and path not in found:pending.append(path)
    while pending:
        p=pending.pop()
        if p in found:continue
        found.add(p);pkg=p.relative_to(ROOT).with_suffix('').parts[:-1]
        for node in ast.walk(ast.parse(p.read_text(encoding='utf-8-sig'))):
            if isinstance(node,ast.Import):
                for n in node.names:add(n.name)
            elif isinstance(node,ast.ImportFrom):
                prefix=list(pkg[:len(pkg)-node.level+1]) if node.level else []
                module='.'.join(prefix+([node.module] if node.module else []));add(module)
                for n in node.names:
                    if n.name!='*':add(module+'.'+n.name)
    found.update([PACKAGE/'job.ps1',PACKAGE/'docs/EXPERIMENT_PLAN.md'])
    return {p.relative_to(ROOT).as_posix():file_digest(p) for p in sorted(found)}


def freeze():
    assert not (PACKAGE/'evidence/selection.json').exists(),'FREEZE_ONCE_ONLY'
    old=read_json(ROOT/'ff_prior_contact_steer/evidence/selection.json')
    stats=read_json(ROOT/'ff_prior_contact_steer/evidence/statistics.json')
    best=min(stats['unit_rows'],key=lambda r:(r['new_best_Q'],r['unit'],r['best_node']))
    source=next(u for u in old['units'] if u['id']==best['unit'])
    tree=OLD/'units'/best['unit']/'tree';chain=best['best_chain'];npz=tree/'nodes.npz'
    with np.load(npz) as a:
        assert all(int(a['parent'][b])==p for p,b in zip(chain[:-1],chain[1:])), 'FROZEN_CHAIN_LINK'
        data={k:a[k][chain] for k in ('q','state','qpos','qvel','elapsed','duration','ee','edge_id')}
        data['parent']=np.arange(len(chain))-1;data['source_node_id']=np.asarray(chain)
    prefix=PACKAGE/'evidence/prefix.npz';atomic_npz(prefix,**data)
    unit=dict(id='u00',task_id=source['task_id'],method='diffusion_geometry_full_horizon',
        seed=source['seed'],proposal_seed=source['proposal_seed'],source_unit=source['id'],
        source_node=best['best_node'],source_node_chain=chain,source_Q=best['new_best_Q'],
        source_prior_candidate=source['source_candidate'],prefix_file=prefix.relative_to(ROOT).as_posix(),
        source_tree=tree.relative_to(ROOT).as_posix(),historical_prefix_level='TREE_SCREEN_ONLY',
        prefix_end_s=float(data['elapsed'][-1]),extension='constant planner q until 27s, optimization guess only',
        repair_candidate_denominator=1,execution_denominator=1)
    # Prior binding supplies transitive public model/task/initial state and neural proposal identities.
    paths={ROOT/p for p in read_json(ROOT/'ff_prior_contact_steer/evidence/binding.json')['files']}
    paths.update([npz,tree/'result.json',prefix,OLD/'run.json',OLD/'physical_contract.json',
        OLD/'scenes'/unit['task_id']/'initial_audit.json',ROOT/'ff_prior_contact_steer/evidence/selection.json',
        ROOT/'ff_prior_contact_steer/evidence/statistics.json',ROOT/'ff_prior_fullchain/evidence/contact_steer_closeout.json',
        PACKAGE/'evidence/authorization.json',PACKAGE/'evidence/budget_derivation.json'])
    # Edges already inherited in the old prefix have their original world-steer binding.
    with np.load(ROOT/source['prefix_file']) as previous:prefix_count=len(previous['q'])
    for node_id,edge in zip(chain,data['edge_id']):
        if node_id>=prefix_count and int(edge)>=0:
            paths.update([tree/'edges'/f'{int(edge):04d}'/'report.json',tree/'edges'/f'{int(edge):04d}'/'trace.npz'])
    entry=next(e for e in old['entries'] if e['sample_id']==unit['task_id']);assert_frozen(entry['task'])
    assert entry['task']['task_hash']==P['task_hash']
    selection=dict(entries=[entry],units=[unit],candidate_denominator=1,execution_denominator=1,
        selection_rule='minimum published contact-steer best_Q then source unit then node id; no new search',
        before_new_physics=True,new_network_samples=0,source_network_steps=16,source_output='102 plus zero221',
        source_condition_dimensions=65,source_network_has_obstacle_condition=False,
        scope='APP_DEV; same original frozen task; no TEST or advantage comparison')
    atomic_json(PACKAGE/'evidence/binding.json',dict(files={p.relative_to(ROOT).as_posix():file_digest(p) for p in sorted(paths)}))
    atomic_json(PACKAGE/'evidence/selection.json',selection)
    return dict(unit=unit,inputs=len(paths),new_physics=0)


def identity():
    binding=read_json(PACKAGE/'evidence/binding.json')
    for p,h in binding['files'].items():
        if file_digest(ROOT/p)!=h:raise ValueError('FROZEN_INPUT_CHANGED: '+p)
    selection=read_json(PACKAGE/'evidence/selection.json')
    assert len(selection['units'])==1 and selection['candidate_denominator']==1
    for e in selection['entries']:assert_frozen(e['task'])
    return dict(protocol=P,source_hashes=sources(),selection=selection,binding=binding)


if __name__=='__main__':
    import json
    print(json.dumps(freeze() if sys.argv[-1]=='freeze' else identity(),ensure_ascii=True))
