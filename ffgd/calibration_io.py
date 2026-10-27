"""Actual trained-checkpoint loading under the planner's data-read sentinel."""
from pathlib import Path
from .architecture_guard import LegacySentinel
from .model_io import load_model
from .orchestration import atomic_json


def run(path,config):
    checkpoint=Path(config['checkpoint'])
    with LegacySentinel([checkpoint,checkpoint.with_suffix('.identity.json')]) as guard:
        model,state,identity=load_model(checkpoint,'cpu')
    report=guard.report();atomic_json(path/'load_guard.json',report)
    return dict(passed=not report['forbidden_calls'] and report['file_audit_enforced'],
                scope='real trained checkpoint loading with strict file audit; not a planning run',checkpoint=identity,
                model_parameters=sum(p.numel() for p in model.parameters()))
