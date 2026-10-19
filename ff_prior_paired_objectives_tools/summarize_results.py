"""Read saved results only. No training, sampling or physics calls."""
from pathlib import Path
import json,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ff_prior.storage import read_json,file_digest
from ff_prior_coverage_train.storage import atomic_json
from ff_prior_paired_objectives.protocol import DEFAULT_OUT,ARMS,ROOT

def summarize(out=DEFAULT_OUT):
    out=Path(out);result={'run':out.name,'status':read_json(out/'status.json'),'additional_physics_calls':0}
    single=read_json(out/'single_step_summary.json');cases=single['cases']
    result['single_step']={'status':single['status'],'cases':len(cases),
        'max_parameter_direction_error':max(x['relative_error'] for c in cases for x in c['parameter_central_checks']),
        'all_three_small_steps_decrease_cases':sum(all(x['loss']<c['before_loss'] for x in c['negative_gradient_steps']) for c in cases),
        'optimizer':{a:{'improved':sum(c['optimizer_steps'][a]['loss']<c['before_loss'] for c in cases),
            'median_ratio':float(np.median([c['optimizer_steps'][a]['loss']/c['before_loss'] for c in cases])),
            'median_parameter_displacement':float(np.median([c['optimizer_steps'][a]['parameter_displacement'] for c in cases]))} for a in ARMS},
        'median_cosines':{k:float(np.median([c['component_gradients']['cosines'][k] for c in cases])) for k in cases[0]['component_gradients']['cosines']},
        'median_weighted_norms':{k:float(np.median([c['component_gradients']['norms'][k] for c in cases])) for k in cases[0]['component_gradients']['norms']}}
    warm=read_json(out/'probes/warm.json')['cases'];result['probes']={};result['training']={}
    for name in ['warm']+list(ARMS):
        path=out/'probes'/f'{name}.json'
        if not path.exists():continue
        rs=read_json(path)['cases'];stats={}
        for t in ('all',500,999):
            group=[(a,b) for a,b in zip(warm,rs) if t=='all' or b['t']==t]
            assert all((a['sample_id'],a['t'],a['seed'])==(b['sample_id'],b['t'],b['seed']) for a,b in group)
            positions=np.array([max(b['position_m'])*1000 for a,b in group]);rotations=np.array([np.rad2deg(max(b['rotation_rad'])) for a,b in group])
            ratios=np.array([b['loss']/max(a['loss'],1e-12) for a,b in group])
            stats[str(t)]={'n':len(group),'loss_ratio_median':float(np.median(ratios)),
                'position_max_arm_median_mm':float(np.median(positions)),
                'rotation_max_arm_median_degrees':float(np.median(rotations)),
                'position_pass':int(sum(positions<=5)),'rotation_pass':int(sum(rotations<=1)),
                'pose_pass':int(sum((positions<=5)&(rotations<=1))),'loss_improved':int(sum(ratios<1))}
        result['probes'][name]=stats
        trpath=out/'learning'/name/'training.json'
        if trpath.exists():
            tr=read_json(trpath);ev=tr['world_records']
            result['training'][name]={k:tr[k] for k in ('status','updates','initial_weight_hash','trained_weight_hash','all_TRAIN_rows_seen','world_gradient_updates','random_sequence_sha256')}
            result['training'][name].update(actual_draws=sum(tr['train_draw_counts'].values()),TRAIN_rows=len(tr['train_draw_counts']),
                gradient_clip_events=sum(e['parameter_gradient_norm']>1 for e in ev),
                event_gradient_norm_median=float(np.median([e['parameter_gradient_norm'] for e in ev])) if ev else None,
                event_gradient_norm_max=max((e['parameter_gradient_norm'] for e in ev),default=None),
                median_component_norms={k:float(np.median([e['components']['norms'][k] for e in ev])) for k in ('v','local','world')} if ev else {},
                median_component_cosines={k:float(np.median([e['components']['cosines'][k] for e in ev])) for k in ('local__v','local__world','v__world')} if ev else {})
    for name in ('TRAIN_selection','training_verification','independent_verification'):
        path=out/f'{name}.json'
        if path.exists():result[name]=read_json(path)
    if (out/'probes/D_both.json').exists():
        final=read_json(out/'probes/D_both.json')['cases']
        gate=lambda c:bool(max(c['position_m'])<=.005 and max(c['rotation_rad'])<=np.deg2rad(1))
        result['D_gate_transitions']={
            'old_pass_lost':sum(gate(a) and not gate(b) for a,b in zip(warm,final)),
            'new_pass_gained':sum(not gate(a) and gate(b) for a,b in zip(warm,final)),
            'both_pass':sum(gate(a) and gate(b) for a,b in zip(warm,final)),
            'position_improved':sum(max(b['position_m'])<max(a['position_m']) for a,b in zip(warm,final)),
            'rotation_improved':sum(max(b['rotation_rad'])<max(a['rotation_rad']) for a,b in zip(warm,final)),
            'loss_improved_but_rotation_worse':sum(b['loss']<a['loss'] and max(b['rotation_rad'])>max(a['rotation_rad']) for a,b in zip(warm,final))}
    result['source_hashes']={p.relative_to(out).as_posix():file_digest(p) for p in [out/'single_step_summary.json',*sorted((out/'probes').glob('*.json')),*sorted((out/'learning').glob('*/training.json'))]}
    return result

if __name__=='__main__':
    r=summarize()
    if '--save' in sys.argv:atomic_json(ROOT/'ff_prior_paired_objectives/evidence/completed_001/posthoc_diagnosis.json',r)
    print(json.dumps(r,ensure_ascii=True))
