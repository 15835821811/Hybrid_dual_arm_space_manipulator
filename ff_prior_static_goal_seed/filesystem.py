"""Validate the actual Windows path envelope before creating experiment output."""
from .protocol import OUT


def worst_relative_paths():
    return [
        'units/u05/candidates/04/repair/iterations/23/trial_06_pairs.json.4294967295.tmp',
        'units/u05/candidates/04/goal_seed/iterations/17/record.json.4294967295.tmp',
        'units/u11/candidates/04/validated/grid_16384_geometry_scans.json.4294967295.tmp',
        'units/u11/candidates/04/validated/grid_16384_numerical.json.4294967295.tmp',
        'scenes/capture_5_obstacles/initial_audit.json.4294967295.tmp',
        'units/u11/execution/geometry_scans.json.4294967295.tmp',
    ]


def probe():
    # A separate non-physics sentinel directory with exactly the same name length.
    directory=OUT.with_name(OUT.name.replace('goal','path'))
    assert len(str(directory))==len(str(OUT))
    assert directory.resolve().parent==OUT.resolve().parent and directory!=OUT
    records=[]
    for rel in worst_relative_paths():
        actual=OUT/rel;sentinel=directory/rel
        if len(str(actual))>=260:raise ValueError('WINDOWS_PATH_ENVELOPE_EXCEEDED: '+str(actual))
        sentinel.parent.mkdir(parents=True,exist_ok=True)
        sentinel.write_text('filesystem probe only; no physical result\n',encoding='utf-8')
        assert sentinel.read_text(encoding='utf-8').startswith('filesystem probe')
        records.append({'relative':rel,'characters':len(str(actual)),'write_read_ok':True})
    return {'status':'PASS','output':str(OUT),'sentinel_directory':str(directory),'records':records,
            'max_characters':max(r['characters'] for r in records),'physics_calls':0}
