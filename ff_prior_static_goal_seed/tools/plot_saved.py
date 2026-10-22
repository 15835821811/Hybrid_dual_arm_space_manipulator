"""Plots saved evidence only; never calls a model, optimizer, or simulator."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

PACKAGE=Path(__file__).resolve().parents[1]
rows=json.loads((PACKAGE/'evidence/candidate_layers.json').read_text(encoding='utf-8'))
stats=json.loads((PACKAGE/'evidence/statistics.json').read_text(encoding='utf-8'))
out=PACKAGE/'docs/figures';out.mkdir(parents=True,exist_ok=True)
fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
names={'structured_geometry':'Structured','diffusion_geometry':'Diffusion'}
colors={'structured_geometry':'#2679b2','diffusion_geometry':'#d16623'}
for method in names:
    selected=[r for r in rows if r['method']==method]
    values=np.array([[r[s]['maximum_position_mm'] for s in ('raw','prepared','seeded')] for r in selected])
    x=np.arange(3)
    for v in values:axes[0].plot(x,v,color=colors[method],alpha=.13,linewidth=.9)
    axes[0].plot(x,np.median(values,axis=0),color=colors[method],marker='o',label=names[method])
axes[0].axhline(5,color='black',linestyle='--',label='5 mm limit')
axes[0].set(yscale='log',xticks=[0,1,2],xticklabels=['Raw','Prepare','World seed'],
    ylabel='Worst-arm endpoint position error (mm)',title='30 proposals: world endpoint recovery')
axes[0].legend(fontsize=9);axes[0].grid(alpha=.2)
for i,unit in enumerate(stats['units']):
    r=next(r for r in rows if r['unit']==unit['unit'] and r['selected_for_repair'])
    y=[r['seeded_clearance_mm'],r['final']['clearance_mm']]
    axes[1].plot([0,1],y,marker='o',color=colors[r['method']],label=f"{names[r['method']]} / {str(r['seed'])[-2:]}")
axes[1].axhline(0,color='black',linestyle='--')
axes[1].set(xticks=[0,1],xticklabels=['Seed screen','Full validation'],ylim=(-75,4),
    ylabel='Minimum signed clearance (mm)',title='6 selected paths: all still penetrate')
axes[1].legend(fontsize=8,ncol=2);axes[1].grid(alpha=.2)
fig.suptitle('static_goal_seed_001: goal recovery is not collision-free planning',fontsize=12)
fig.savefig(out/'endpoint_and_clearance.png',dpi=180)
fig.savefig(out/'endpoint_and_clearance.pdf')
plt.close(fig)
print(out/'endpoint_and_clearance.png')
