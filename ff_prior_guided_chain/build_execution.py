"""One-time source adaptation; copied gate formulas remain unchanged."""
from pathlib import Path
from .protocol import ROOT,PACKAGE


def copy(name,source,replacements):
    dest=PACKAGE/name
    text=(ROOT/source).read_text(encoding='utf-8-sig')
    for old,new,count in replacements:
        if text.count(old)!=count:raise ValueError('SOURCE_ADAPTER_COUNT:'+name+':'+old)
        text=text.replace(old,new)
    if dest.exists():
        assert dest.read_text(encoding='utf-8')==text,'ADAPTER_ALREADY_EDITED:'+name
    else:dest.write_text(text,encoding='utf-8')


def main():
    copy('static_tracker.py','ff_prior_balanced_restore/tracker.py',[
        ('from .native_budget import','from ff_prior_balanced_restore.native_budget import',1),
        ('data = initial.reset(robot)','data = robot._guided_actual_data\n    assert np.array_equal(integration_state(robot,data),initial.values)',1)])
    copy('static_audit.py','ff_prior_balanced_restore/physical_audit.py',[
        ('from .tracker import','from .static_tracker import',2),
        ('from .native_budget import','from ff_prior_balanced_restore.native_budget import',1),
        ('    data = initial.reset(robot)\n    recorded=',
         '    data = robot._guided_replay_data\n    assert np.array_equal(integration_state(robot,data),initial.values)\n    recorded=',1)])
    copy('dynamic_environment.py','ff_prior_balanced_brake/environment.py',[
        ('from .protocol import P','from ff_prior_balanced_brake.protocol import P',1),
        ('from .reference import bypass','from ff_prior_balanced_brake.reference import bypass',1),
        ('from .controller_observation import ObservedWholeBodyFilter','from ff_prior_balanced_brake.controller_observation import ObservedWholeBodyFilter',1),
        ('self.data=self.initial.reset(robot)','self.data=robot._guided_actual_data',1)])
    copy('dynamic_evaluation.py','ff_prior_balanced_brake/evaluation.py',[
        ('from .protocol import P','from ff_prior_balanced_brake.protocol import P',1),
        ('replay=env.initial.reset(r)','replay=r._guided_replay_data\n    assert np.array_equal(integration(r,replay),env.initial.values)',1)])
    copy('satellite_environment.py','ff_prior_satellite_origin/environment.py',[
        ('from ff_prior_balanced_brake.environment import Environment as PreviousEnvironment','from .dynamic_environment import Environment as PreviousEnvironment',1),
        ('from .protocol import P','from ff_prior_satellite_origin.protocol import P',1),
        ('from .force import','from ff_prior_satellite_origin.force import',1),
        ('from .diagnostics import','from ff_prior_satellite_origin.diagnostics import',1),
        ('from .compact_storage import','from ff_prior_satellite_origin.compact_storage import',1)])
    copy('satellite_evaluation.py','ff_prior_satellite_origin/evaluation.py',[
        ('from .protocol import P','from ff_prior_satellite_origin.protocol import P',1),
        ('from .ledger import','from ff_prior_satellite_origin.ledger import',1),
        ('replay=env.initial.reset(r)','replay=r._guided_replay_data\n    assert np.array_equal(integration(r,replay),env.initial.values)',1)])


if __name__=='__main__':main()
