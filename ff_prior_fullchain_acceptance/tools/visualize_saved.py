"""Plot final evidence and build an offline recorded-endpoint viewer."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
PACKAGE=Path(__file__).resolve().parents[1]
s=json.loads((PACKAGE/'evidence/statistics.json').read_text(encoding='utf-8'))
paths=json.loads((PACKAGE/'evidence/native_endpoint_paths.json').read_text(encoding='utf-8'))
out=PACKAGE/'docs/figures';out.mkdir(parents=True,exist_ok=True)
fig,axes=plt.subplots(1,2,figsize=(12,4.6),constrained_layout=True)
x=np.arange(5);counts=[6,0,6,6,0];blocked=[0,6,0,0,6]
axes[0].bar(x,counts,color='#307d8c',label='Verified logical opportunities')
axes[0].bar(x,blocked,bottom=counts,color='#dddddd',hatch='//',label='Dependency blocked')
axes[0].set(xticks=x,xticklabels=['Static 3','Static 5','Stationary','Drifting','5 → drift'],ylim=(0,7.5),ylabel='Logical opportunities / 6',title='3 / 5 scene types covered; full chain incomplete')
for i in x:axes[0].text(i,6.2,'NEW' if i==0 else ('REUSE' if i in [2,3] else 'BLOCKED'),ha='center',fontsize=8)
axes[0].legend(fontsize=8,loc='upper center',bbox_to_anchor=(.5,-.12))
for method,color,label in [('deterministic_system','#2679b2','Structured'),('diffusion_system','#d16623','Diffusion + common correction')]:
    rows=sorted([r for r in s['rows'] if r['new_execution'] and r['method']==method],key=lambda r:r['seed'])
    axes[1].plot([0,1,2],[r['hold_position_mm']['maximum'] for r in rows],marker='o',color=color,label=label)
axes[1].axhline(5,color='black',linestyle='--',label='5 mm limit')
axes[1].set(xticks=[0,1,2],xticklabels=['20264611','20264617','20264623'],ylabel='Worst hold position error (mm)',title='New static executions: same frozen public references',ylim=(-.15,5.6))
axes[1].legend(fontsize=8);axes[1].grid(alpha=.2)
fig.savefig(out/'coverage_and_hold.png',dpi=180);fig.savefig(out/'coverage_and_hold.pdf');plt.close(fig)
html='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>P4 已记录双末端轨迹</title>
<style>body{font:16px system-ui;margin:24px;background:#f5f7fa;color:#203345}main{max-width:1100px;margin:auto}canvas{background:white;width:100%;border:1px solid #ccd7df}select,input,button{font:inherit;margin:6px;padding:5px}.note{line-height:1.6}code{font-size:12px;word-break:break-all}</style>
<main><h1>P4：已记录的三障碍双末端路径</h1><p class="note">仅显示真实原生执行保存的末端位置投影，每100ms取样。不是重新仿真，不是全机碰撞可视证明；资格来自完整几何验收和836维状态独立重放。五障碍及连续组合未通过，不在此伪造轨迹。</p>
<label>执行记录<select id="unit"></select></label><label>投影<select id="plane"><option value="0,1">XY</option><option value="0,2">XZ</option><option value="1,2">YZ</option></select></label>
<button id="play">播放 / 暂停</button><input id="frame" type="range" min="0" value="0"><span id="clock"></span><canvas id="chart" width="1000" height="520"></canvas>
<p>蓝色：连续体末端；橙色：刚性臂末端。十字：冻结终点。</p><p><code id="source"></code></p><p><code id="hash"></code></p></main>
<script>const records=__DATA__; const unit=document.getElementById('unit'),plane=document.getElementById('plane'),slider=document.getElementById('frame'),cv=document.getElementById('chart'),ctx=cv.getContext('2d');let playing=false;
records.forEach((r,i)=>{const o=document.createElement('option');o.value=i;o.textContent=r.id+' — '+r.method;unit.append(o)});
function draw(){const r=records[+unit.value],ax=plane.value.split(',').map(Number);slider.max=r.ee.length-1;let k=Math.min(+slider.value,r.ee.length-1);slider.value=k;const pts=r.ee.flat().concat(r.goals);let lo=ax.map(a=>Math.min(...pts.map(p=>p[a]))),hi=ax.map(a=>Math.max(...pts.map(p=>p[a])));const scale=Math.min(880/Math.max(hi[0]-lo[0],.01),400/Math.max(hi[1]-lo[1],.01));const mid=lo.map((v,i)=>(v+hi[i])/2);function xy(p){return[500+(p[ax[0]]-mid[0])*scale,260-(p[ax[1]]-mid[1])*scale]};ctx.clearRect(0,0,1000,520);ctx.font='14px sans-serif';ctx.fillStyle='#384e60';ctx.fillText('世界坐标 / m；同一投影等比例',25,25);
['#2679b2','#d16623'].forEach((color,a)=>{ctx.strokeStyle=color;ctx.lineWidth=1;ctx.globalAlpha=.22;ctx.beginPath();r.ee.forEach((poses,i)=>{const p=xy(poses[a]);i?ctx.lineTo(...p):ctx.moveTo(...p)});ctx.stroke();ctx.globalAlpha=1;ctx.lineWidth=3;ctx.beginPath();for(let j=0;j<=k;j++){const p=xy(r.ee[j][a]);j?ctx.lineTo(...p):ctx.moveTo(...p)}ctx.stroke();let p=xy(r.ee[k][a]);ctx.fillStyle=color;ctx.beginPath();ctx.arc(...p,5,0,2*Math.PI);ctx.fill();p=xy(r.goals[a]);ctx.beginPath();ctx.moveTo(p[0]-7,p[1]);ctx.lineTo(p[0]+7,p[1]);ctx.moveTo(p[0],p[1]-7);ctx.lineTo(p[0],p[1]+7);ctx.stroke()});document.getElementById('clock').textContent=r.time[k].toFixed(2)+' s';document.getElementById('source').textContent=r.source;document.getElementById('hash').textContent='SHA256 '+r.sha256;}
unit.onchange=()=>{slider.value=0;draw()};plane.onchange=draw;slider.oninput=draw;document.getElementById('play').onclick=()=>playing=!playing;setInterval(()=>{if(playing){slider.value=(+slider.value+1)%(+slider.max+1);draw()}},100);draw();</script></html>'''
(PACKAGE/'docs/recorded_endpoints.html').write_text(html.replace('__DATA__',json.dumps(paths,separators=(',',':'))),encoding='utf-8')
print(out/'coverage_and_hold.png')
