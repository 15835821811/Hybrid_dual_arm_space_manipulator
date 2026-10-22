"""Bind exactly three public-search prefixes; never regenerate old physics."""
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
    target=PACKAGE/'evidence/selection.json'
    if target.exists():raise ValueError('FREEZE_IS_ONCE_ONLY')
    old=read_json(ROOT/'ff_prior_world_steer/evidence/selection.json')
    statistics=read_json(ROOT/'ff_prior_world_steer/evidence/candidate_statistics.json')
    receipt=ROOT/'ff_prior_fullchain/evidence/world_steer_closeout.json'
    assert read_json(receipt)['closed']
    paths={ROOT/p for p in read_json(ROOT/'ff_prior_world_steer/evidence/binding.json')['files']}
    paths.update([receipt,OLD/'run.json',OLD/'physical_contract.json',
        ROOT/'ff_prior_world_steer/evidence/selection.json',
        ROOT/'ff_prior_world_steer/evidence/candidate_statistics.json',
        ROOT/'ff_prior_world_steer/evidence/completed_resume.json',
        ROOT/'paper/review-traces/experiment-result-to-claim/20261022_world_steer/verdict.json',
        PACKAGE/'evidence/authorization.json'])
    units=[]
    for source in old['units']:
        best=min((r for r in statistics if r['unit']==source['id']),key=lambda r:(r['best_Q'],r['candidate'],r['best_node']))
        candidate=OLD/'units'/source['id']/'candidates'/f'{best["candidate"]:02d}'
        directory=candidate/'tree';nodes_path=directory/'nodes.npz'
        with np.load(nodes_path) as n:
            chain=[];i=int(best['best_node'])
            while i>=0:chain.append(i);i=int(n['parent'][i])
            chain.reverse();assert chain[0]==0
            prefix={k:n[k][chain] for k in ('q','state','qpos','qvel','elapsed','duration','ee','edge_id')}
            prefix['parent']=np.arange(len(chain))-1
            prefix['source_node_id']=np.asarray(chain)
            assert len(prefix['state'][0])==836 and prefix['elapsed'][0]==0
            prefix_path=PACKAGE/'evidence'/f'prefix_{source["id"]}.npz'
            atomic_npz(prefix_path,**prefix)
            for edge in prefix['edge_id'][1:]:
                paths.update([directory/'edges'/f'{int(edge):04d}'/'report.json',directory/'edges'/f'{int(edge):04d}'/'trace.npz'])
        paths.update([nodes_path,directory/'result.json',candidate/'controls.npz',candidate/'provenance.json',prefix_path,
            OLD/'scenes'/source['task_id']/'initial_audit.json',OLD/'scenes'/source['task_id']/'limits.json'])
        units.append(dict(id=source['id'],task_id=source['task_id'],method='diffusion_geometry',
            seed=source['seed'],proposal_seed=source['proposal_seeds'][best['candidate']],
            source_candidate=best['candidate'],source_node=best['best_node'],source_node_chain=chain,
            source_tree=directory.relative_to(ROOT).as_posix(),
            prefix_file=prefix_path.relative_to(ROOT).as_posix(),prefix_hash=file_digest(prefix_path),
            historical_prefix_level='TREE_SCREEN_ONLY',remaining_time=27.-float(prefix['elapsed'][-1]),
            source_Q=best['best_Q'],repair_candidate_denominator=1,execution_denominator=1))
    selection=dict(entries=old['entries'],units=units,candidate_denominator=3,execution_denominator=3,
        selection_rule='per original seed group: lowest saved best_Q, candidate id, node id',
        before_new_physics=True,new_neural_samples=0,historical_model_prefixes=3,
        actual_execution_from_original_initial_state=True,advantage_comparison=False)
    for e in selection['entries']:assert_frozen(e['task'])
    atomic_json(PACKAGE/'evidence/binding.json',{'files':{p.relative_to(ROOT).as_posix():file_digest(p) for p in sorted(paths)}})
    atomic_json(target,selection)
    return dict(units=units,bound_inputs=len(paths),new_physics=0)


def identity():
    binding=read_json(PACKAGE/'evidence/binding.json')
    for p,h in binding['files'].items():
        if file_digest(ROOT/p)!=h:raise ValueError('FROZEN_INPUT_CHANGED: '+p)
    selection=read_json(PACKAGE/'evidence/selection.json')
    assert len(selection['units'])==3 and selection['candidate_denominator']==3
    for e in selection['entries']:assert_frozen(e['task'])
    return dict(protocol=P,source_hashes=sources(),selection=selection,binding=binding)


if __name__=='__main__':
    import json
    print(json.dumps(freeze() if sys.argv[-1]=='freeze' else identity(),ensure_ascii=True))
