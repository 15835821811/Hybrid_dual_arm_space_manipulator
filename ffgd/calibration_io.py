"""Actual trained-checkpoint loading under the planner's data-read sentinel."""
from pathlib import Path
import json
from .architecture_guard import LegacySentinel
from .orchestration import atomic_json


def run(path,config):
    checkpoint=Path(config['checkpoint'])
    allowed=[checkpoint,checkpoint.with_suffix('.identity.json')]
    if config.get('task_file'):
        task_file=Path(config['task_file']);submitted=json.loads(task_file.read_text(encoding='utf-8-sig'));allowed.append(task_file)
        if 'snapshot_file' in submitted.get('initial',{}):allowed.append(task_file.parent/submitted['initial']['snapshot_file'])
    problem_loaded=False
    with LegacySentinel(allowed) as guard:
        if config.get('task_file'):
            from .problem_io import load_problem
            robot,snapshot,task,problem=load_problem(task_file);problem_loaded=True
        from .model_io import load_model
        model,state,identity=load_model(checkpoint,config.get('device','cpu'))
    report=guard.report();atomic_json(path/'load_guard.json',report)
    return dict(passed=not report['forbidden_calls'] and report['file_audit_enforced'],
                scope='real public input/robot/checkpoint loading with strict file audit; not a planning run',checkpoint=identity,
                public_problem_loaded=problem_loaded,
                device=config.get('device','cpu'),
                model_parameters=sum(p.numel() for p in model.parameters()))
