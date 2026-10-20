import ast
from pathlib import Path
from ff_prior.storage import read_json,file_digest,digest
from .protocol import ROOT,SOURCE,CONTROL,WARM,BASE,PROTOCOL

def sources():
    pending = list((ROOT/'ff_prior_smooth_endpoint').glob('*.py'))
    found = set()
    def add(module):
        parts = module.split('.')
        for i in range(1, len(parts)+1):
            p = ROOT.joinpath(*parts[:i])
            for path in (p.with_suffix('.py'), p/'__init__.py'):
                if path.is_file() and path not in found:
                    pending.append(path)
    while pending:
        path = pending.pop()
        if path in found:
            continue
        found.add(path)
        parts = path.relative_to(ROOT).with_suffix('').parts
        package = parts[:-1]
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8-sig'))):
            if isinstance(node, ast.Import):
                for name in node.names:
                    add(name.name)
            elif isinstance(node, ast.ImportFrom):
                prefix = list(package[:len(package)-node.level+1]) if node.level else []
                module = '.'.join(prefix+([node.module] if node.module else []))
                add(module)
                for name in node.names:
                    if name.name != '*':
                        add(module+'.'+name.name)
    # Non-Python launcher is also frozen, but mutable goal/tracker are not code.
    found.add(ROOT/'ff_prior_smooth_endpoint/job.ps1')
    return {p.relative_to(ROOT).as_posix(): file_digest(p) for p in sorted(found)}



def identity():
    from ff_prior_expand256_train.inputs import identity_for
    if identity_for()[0]!=read_json(CONTROL/'run.json')['identity']:raise ValueError('control identity changed')
    if read_json(CONTROL/'independent_verification.json')['status']!='PASS':raise ValueError('control audit not PASS')
    binding=read_json(ROOT/'ff_prior_smooth_endpoint/evidence/binding.json')
    for path,h in binding['files'].items():
        if file_digest(ROOT/path)!=h:raise ValueError('bound control file changed: '+path)
    return {'protocol':PROTOCOL,'source_hashes':sources(),'binding':binding,
            'selection':read_json(ROOT/'ff_prior_smooth_endpoint/evidence/selection.json')}

def freeze():
    entries=read_json(ROOT/'ff_prior_expand256/evidence/target_selection.json')['entries']
    diagnostic=[];event=[]
    for obstacle in (False,True):
        pool=sorted([e for e in entries if bool(e['scene']['objects'])==obstacle],key=lambda e:e['task']['task_hash'])
        diagnostic.extend(pool[:4]);event.extend(pool[4:20])
    def short(e):return {'sample_id':e['parent_sample_id'],'task_hash':e['task']['task_hash'],'obstacle':bool(e['scene']['objects'])}
    selection={'diagnostic':[short(e) for e in diagnostic],'events':[short(e) for e in event],'TRAIN_only':True,'frozen_before_run':True,
      'cases':[{'target':i,'t':t,'noise_seed':PROTOCOL['diagnostic_seed']+i*31+j,'control_target':(i+1)%8} for i in range(8) for j,t in enumerate(PROTOCOL['noise_times'])],
      'world_sequence':[{'update':100*(j+1),'targets':[(2*j)%32],'control_targets':[(2*j+1)%32],
        't':PROTOCOL['noise_times'][j%3],'noise_seeds':[PROTOCOL['event_seed']+j*11+k for k in range(2)]} for j in range(60)]}
    files={}
    for directory,names in [(CONTROL,['run.json','summary.json','independent_verification.json','frozen_tasks.json','physical_contract.json','training_index.json','training_input_audit.json','loaded_training_arrays.npz','sensitivity_summary.json','learning/weighted_mean.json','learning/goal_loss/model.pt','learning/goal_loss/codec.json','learning/goal_loss/training.json']),
                            (WARM,['learning/goal_loss/model.pt','learning/goal_loss/codec.json','learning/goal_loss/training.json'])]:
        for name in names:files[(directory/name).relative_to(ROOT).as_posix()]=file_digest(directory/name)
    # All reused raw candidates, selections, outcomes, reports and native traces are immutable.
    for method in PROTOCOL['reuse_methods'].values():
        for directory in ['candidates','executions']:
            for p in (CONTROL/directory).glob('*/*/'+method+'/*'):
                if p.is_file():files[p.relative_to(ROOT).as_posix()]=file_digest(p)
    from ff_prior_pilot_train.storage import atomic_json
    atomic_json(ROOT/'ff_prior_smooth_endpoint/evidence/selection.json',selection)
    atomic_json(ROOT/'ff_prior_smooth_endpoint/evidence/binding.json',{'files':files,'controls_reused_not_rerun':True})
    return {'diagnostic_targets':len(diagnostic),'event_targets':len(event),'bound_files':len(files)}
