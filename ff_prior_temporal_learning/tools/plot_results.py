"""Plot saved trajectories only; never generate or replay physics."""
import csv
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from ff_prior.coupled_predictor import pose_errors

run=ROOT/'artifacts/v7_1/temporal_learning_001'
out=ROOT/'ff_prior_temporal_learning/evidence'
stats=json.loads((out/'completed_statistics.json').read_text(encoding='utf-8'))
fig,axes=plt.subplots(2,2,figsize=(12,7.5),layout='constrained')
for ax,units,title in zip(axes.ravel(),[
    ['d000','d001','d011','d021'],['d100','d101','d111','d121'],
    ['v00','v10','v20','v30'],['v01','v11','v21','v31']],
    ['DEV stationary: common feedback','DEV drifting: common feedback',
     'VAL deterministic: stopped at penetration','VAL Diffusion: no world feedback']):
    for unit in units:
        with np.load(run/unit/'native_trace.npz') as a:
            pos,_=pose_errors(a['ee'],a['desired'])
            label=('Deterministic' if unit in ['d000','d100'] else unit)
            ax.plot(a['time'][::10],pos.max(axis=1)[::10]*1000,label=label,lw=1.3)
            if unit.startswith('v') and unit.endswith('0'):
                ax.plot(a['time'][-1],pos.max(axis=1)[-1]*1000,'x',color=ax.lines[-1].get_color())
    ax.set(title=title,xlabel='Time (s)',ylabel='Worst-arm position error (mm)',xlim=(0,27))
    if title.startswith('VAL'): ax.axhline(5,color='black',ls='--',lw=.7,label='5 mm gate')
    ax.grid(alpha=.25);ax.legend(fontsize=8)
fig.suptitle('P3b temporal_learning_001 | raw saved native traces | no new simulation')
fig.savefig(out/'world_position_traces.png',dpi=150)
plt.close(fig)
with (out/'unit_metrics.csv').open('w',newline='',encoding='utf-8-sig') as f:
    w=csv.writer(f);w.writerow(['unit','split','method','scene','new_execution','status','duration_s','position_mean_mm','position_p95_mm','position_max_mm','rotation_max_deg','prepare_scale','failed_gates'])
    for r in stats['units']:
        w.writerow([r['unit'],r['split'],r['method'],r.get('scene',''),r['new_execution'],r['status'],r['observed_duration_s'],r['position_mm']['mean'],r['position_mm']['p95'],r['position_mm']['maximum'],r['rotation_deg']['maximum'],r['prepare']['scale'],';'.join(r['failed_gates'])])
print(str(out/'world_position_traces.png'))
