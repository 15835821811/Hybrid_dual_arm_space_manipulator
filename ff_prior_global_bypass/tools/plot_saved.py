"""Scientific figure from saved statistics; no robot or new simulation."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PACKAGE=Path(__file__).resolve().parents[1]
s=json.loads((PACKAGE/'evidence/statistics.json').read_text(encoding='utf-8'))
out=PACKAGE/'docs/figures';out.mkdir(parents=True,exist_ok=True)
fig,axes=plt.subplots(1,3,figsize=(14,4.5),constrained_layout=True)
methods=[('structured_geometry','Structured','#2679b2'),('diffusion_geometry','Diffusion bias','#d16623')]
seeds=[20264611,20264617,20264623];x=np.arange(3)
for index,(method,label,color) in enumerate(methods):
    rows=[next(u for u in s['units'] if u['method']==method and u['seed']==seed) for seed in seeds]
    axes[0].bar(x+(index-.5)*.34,[r['nodes'] for r in rows],width=.32,label=label,color=color)
    axes[1].plot(x,[r['best_world_Q']*5 for r in rows],'-o',color=color,label=label)
axes[0].set(xticks=x,xticklabels=['Seed 11','Seed 17','Seed 23'],ylabel='Accepted exploration nodes',
    title='2,470 tree nodes; none are full trajectories')
axes[0].legend(fontsize=8)
axes[1].axhline(5,color='#7f2525',linestyle='--',label='5 mm position gate')
axes[1].set(xticks=x,xticklabels=['Seed 11','Seed 17','Seed 23'],ylabel='Best prefix worst-arm position / mm',
    title='World target remains unreached',ylim=(0,130))
axes[1].legend(fontsize=8)
names=['Seed pose\nonly','Complete\ntree route','MODEL\nVALID','Native +\nreplay']
axes[2].bar(np.arange(4),[30,0,0,0],color=['#8796a6','#bbbbbb','#bbbbbb','#bbbbbb'])
for i,v in enumerate([30,0,0,0]):axes[2].text(i,v+.8,str(v),ha='center')
axes[2].set(xticks=np.arange(4),xticklabels=names,ylim=(0,35),ylabel='Count',title='Downstream stages not run')
fig.suptitle('Original five-obstacle development task: bounded global search did not produce a qualified path',fontsize=12)
fig.savefig(out/'global_bypass_summary.png',dpi=160)
fig.savefig(out/'global_bypass_summary.pdf')
plt.close(fig)
