"""Pure numerical/schema checks; no MuJoCo model, geometry query, or native step."""
import ast
import json
import re
import shutil
import subprocess
from pathlib import Path
import tempfile
import numpy as np
from ff_prior_pilot_train.storage import atomic_json, atomic_npz
from .diagnostics import discrete_curvature
from .media import frame_segments, HTML
from .plots import BodySeries, time_cell_edges


def run():
    checks = []
    line = np.column_stack([np.arange(30), np.zeros((30, 2))])
    assert np.array_equal(discrete_curvature(line), np.zeros(28))
    checks.append('straight_centerline_zero_curvature')
    theta = np.linspace(0, np.pi/2, 30)
    circle = 2*np.column_stack([np.cos(theta), np.sin(theta), np.zeros(30)])
    assert np.allclose(discrete_curvature(circle), .5, atol=.0001)
    checks.append('circle_centerline_expected_curvature')
    mixed_times = np.array([0., .002, .004, .024, .044])
    edges = time_cell_edges(mixed_times)
    assert np.allclose(edges, [0., .001, .003, .014, .034, .044])
    assert np.isclose(np.diff(edges).sum(), .044)
    assert np.isclose(edges[-2]-edges[-3], .020)
    checks.append('mixed_2ms_20ms_heatmap_uses_actual_nonuniform_time_cells')
    pose = np.eye(4); pose[:3, 3] = [1, 2, 3]
    for axis, (a, b) in enumerate(frame_segments(pose, .07)):
        assert np.allclose(b-a, .07*np.eye(3)[axis]) and np.array_equal(a, [1, 2, 3])
    checks.append('SE3_frame_axis_direction_and_scale')
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        meta = dict(status='COMPLETE', samples=2, expected_samples=2,
            links=[dict(body_id=1, body_name='fixture_link', arm=0)],
            obstacles=[dict(object_id='fixture_obstacle')], geom_names=['fixture_geom'])
        for name, times in [('a', [0., 1.]), ('b', [1., 2.])]:
            directory = root/name; directory.mkdir()
            atomic_json(directory/'body_obstacle_metadata.json', meta)
            atomic_npz(directory/'body_obstacle_trace.npz', time=times,
                distance_m=np.ones((2, 1, 1)), nearest_geom_id=np.zeros((2, 1, 1), dtype=int),
                connector=np.zeros((2, 1, 1, 2, 3)))
        series = BodySeries([root/'a', root/'b'])
        assert np.array_equal(series.a['time'], [0., 1., 2.])
        assert series.frame(.5) is None and series.frame(1.)['nearest'][0]['body_name'] == 'fixture_link'
        checks.append('exact_tick_only_no_interpolated_distance_and_single_handoff_row')
        missing = BodySeries([root/'a', root/'missing'])
        assert not missing.complete
        checks.append('missing_diagnostics_cannot_be_complete')
    for path in Path(__file__).parent.glob('*.py'):
        tree = ast.parse(path.read_text(encoding='utf-8'))
        assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                       and n.func.attr == 'mj_step' for n in ast.walk(tree))
    checks.append('all_shared_modules_have_no_native_step_calls')
    assert 'for(let k=1;k<=i;k++)' in HTML and 'D.goal_poses[i][arm]' in HTML
    checks.append('cumulative_actual_paths_and_synchronized_goal_frames')
    node = shutil.which('node')
    if not node:
        raise RuntimeError('NODE_REQUIRED_FOR_INTERACTIVE_SOFTWARE_CHECK')
    fixture = dict(time=[0., .002], bodies=[[[0, 0, 0], [1, 0, 0]]]*2,
        parents=[0, 0], waypoints=[[1, k*.02, 0] for k in range(7)], obstacles=[],
        ee=[[[1, 0, 0], [1, .2, 0]]]*2, ee_poses=[[pose.tolist(), pose.tolist()]]*2,
        goal_poses=[[pose.tolist(), pose.tolist()]]*2, body_frames=[None, None],
        position_mm=[[0, 0], [0, 0]], clearance_mm=[None, .1])
    script = re.search(r'<script>(.*)</script>', HTML, re.S).group(1)
    script = script.replace('__DATA__', json.dumps(fixture))
    harness = '''const vm=require('vm');const elements={};const context={
      beginPath(){},moveTo(...v){if(v.some(x=>!Number.isFinite(x)))throw Error('nonfinite canvas')},
      lineTo(){},stroke(){},setLineDash(){},arc(){},fill(){},fillText(){},clearRect(){}};
      const sandbox={devicePixelRatio:1,window:{},requestAnimationFrame(){},
        document:{getElementById(id){return elements[id]??=(id==='c'?{clientWidth:900,getContext(){return context},setPointerCapture(){}}:{})}}};
      vm.runInNewContext(SCRIPT,sandbox);
      elements.seek.value=1;elements.seek.oninput();
      if(!elements.info.textContent.includes('0.002'))throw Error('seek draw failed');'''
    result = subprocess.run([node, '-e', harness.replace('SCRIPT', json.dumps(script))],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    checks.append('interactive_initial_draw_and_seek_synthetic_DOM')
    return dict(status='PASS', checks=checks, new_native_steps=0, new_coupled_calls=0,
        new_geometry_evaluations=0, synthetic_fixtures_only=True)


if __name__ == '__main__':
    print(json.dumps(run(), ensure_ascii=False))
