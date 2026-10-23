"""Build a self-contained draft geometry page; never starts a physics worker."""
import json,socket,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];E=ROOT/'ff_prior_balanced_scene/evidence'
d=json.loads((E/'preview_data.json').read_text(encoding='utf-8'))
layout=json.loads((E/'layout.json').read_text(encoding='utf-8'))
rows=''.join('<tr><td>'+o['object_id']+'</td><td>'+', '.join(f'{v:.6f}' for v in o['position'])+'</td><td>'+str(round(o['size'][0]*1000))+'</td></tr>' for o in d['objects'])
html='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>双侧七障碍 · 布局草案</title>
<meta name="viewport" content="width=device-width,initial-scale=1"><style>
body{margin:0;background:#101923;color:#e2eaf2;font:16px system-ui}main{max-width:1200px;margin:auto;padding:28px}h1{font-size:30px}h2{font-size:21px}p{line-height:1.7}section{padding:22px;margin:20px 0;background:#192635;border:1px solid #34475c;border-radius:12px}.tag{color:#ffd38c}.legend{display:flex;gap:20px;flex-wrap:wrap}.legend span{padding:8px}canvas{display:block;width:100%;height:360px;touch-action:none;background:#101923}button{background:#2c425b;border:1px solid #607893;border-radius:6px;color:#fff;padding:8px 18px;margin:8px 6px 8px 0}img{width:100%}table{width:100%;border-collapse:collapse}td,th{border-bottom:1px solid #42546a;text-align:left;padding:12px}a{color:#70d6f4}.small{color:#b4c4d4;font-size:14px}</style>
<main><h1>刚性侧 2 个 · 连续体侧 5 个</h1><p class="tag">布局预览，尚未执行新场景。骨架为完整初态；蓝色虚线只连接原七航点。</p>
<section><h2>看清末端通道与臂身空间</h2><p>R1/R2 疏散布置，球面相距 0.626 m。C1–C3 位于 W1→W2、W3→W4、W5→W6 段；C4/C5 分布在臂身中段与中远段。拖动旋转、滚轮缩放；投影重合不代表三维重合。</p>
<div class="legend"><span style="color:#eda073">● 刚性 R1/R2</span><span style="color:#5dd58a">● 末端段 C1–C3</span><span style="color:#bb91f3">● 臂身 C4/C5</span><span style="color:#66c6fa">W1–W7 原不规则航点</span></div>
<canvas id="c"></canvas><button id="oblique">三维</button><button id="top">俯视 XY</button><button id="side">航点 YZ</button><p class="small">没有新执行轨迹或成功动画。当前仅一次固定初态全机几何检查通过；不能据此认定整条运动可行。</p></section>
<section><h2>固定的球心与尺寸</h2><table><tr><th>障碍</th><th>世界球心 XYZ（m）</th><th>半径（mm）</th></tr>__ROWS__</table><p>保持原七点的准确坐标、姿态与各段到达时间。新增臂身距离热图、最近臂段高亮及约束活跃记录，用新运行证据判断臂形的实际作用。</p></section>
<section><h2>一次新场景连续工程单元</h2><p>从完整初态重新开始：27 秒接近 + 0.5 秒保持，经模型／执行／836 完整重放通过后，从同场景真实末态接 27 秒七航点动态。障碍不在交接时改变。</p><p>预算草案：至多 28 轮恢复，13,000 耦合，54,500 原生步（含两段完整重放），4.5 小时，14 GiB 输出及 1 GiB 收尾，均在原周期剩余额度内。无第二候选、额外修复、训练、扩库、对照或 TEST。</p><p class="tag">当前只批准了布局选择；新增完整初态运行范围待确认。旧原五障碍证据独立保留。</p><p><a href="layout.json">精确布局与固定初态检查</a> · <a href="layout.pdf">布局PDF</a> · <a href="LAYOUT_AND_BUDGET.md">完整范围与预算</a> · <a href="http://127.0.0.1:8769/index.html">原五障碍实际执行媒体</a></p></section>
<section><h2>三方向静态示意</h2><img src="layout.png"></section></main>
<script>const D=__DATA__,c=document.getElementById('c'),ctx=c.getContext('2d');let yaw=-.6,pitch=.5,zoom=2.3,drag=null;const center=[1.05,-.14,0],span=2.3;
function project(p){let x=p[0]-center[0],y=p[1]-center[1],z=p[2]-center[2],a=x*Math.cos(yaw)-y*Math.sin(yaw),b=x*Math.sin(yaw)+y*Math.cos(yaw),scale=Math.min(c.width,c.height)*.84/span*zoom;return[c.width/2+a*scale,c.height/2-(z*Math.cos(pitch)-b*Math.sin(pitch))*scale,b*Math.cos(pitch)+z*Math.sin(pitch),scale]}
function line(a,b,color,width=2,dash=false){a=project(a);b=project(b);ctx.strokeStyle=color;ctx.lineWidth=width*devicePixelRatio;ctx.setLineDash(dash?[6*devicePixelRatio,5*devicePixelRatio]:[]);ctx.beginPath();ctx.moveTo(a[0],a[1]);ctx.lineTo(b[0],b[1]);ctx.stroke();ctx.setLineDash([])}
function label(p,s,color,dy=-9){let q=project(p);ctx.fillStyle=color;ctx.font=(14*devicePixelRatio)+'px system-ui';ctx.fillText(s,q[0]+5*devicePixelRatio,q[1]+dy*devicePixelRatio)}
function draw(){c.width=c.clientWidth*devicePixelRatio;c.height=360*devicePixelRatio;ctx.clearRect(0,0,c.width,c.height);for(let i=1;i<D.bodies.length;i++){let p=D.parents[i];if(p>0)line(D.bodies[p],D.bodies[i],i<65?'#4cd2de':'#ffbb69',3)}for(let k=1;k<D.points.length;k++)line(D.points[k-1],D.points[k],'#66c6fa',2,true);D.points.forEach((p,k)=>{let q=project(p);ctx.fillStyle='#66c6fa';ctx.beginPath();ctx.arc(q[0],q[1],4*devicePixelRatio,0,7);ctx.fill();label(p,'W'+(k+1),'#a4dcff')});[...D.objects].sort((a,b)=>project(a.position)[2]-project(b.position)[2]).forEach(o=>{let q=project(o.position),color=o.object_id[0]==='R'?'#eda073':(['C4','C5'].includes(o.object_id)?'#bb91f3':'#5dd58a');ctx.fillStyle=color+'60';ctx.strokeStyle=color;ctx.lineWidth=2*devicePixelRatio;ctx.beginPath();ctx.arc(q[0],q[1],Math.max(4*devicePixelRatio,o.size[0]*q[3]),0,7);ctx.fill();ctx.stroke();label(o.position,o.object_id,color,19)});let origin=[.1,-1.05,-.2];for(let k=0;k<3;k++){let end=origin.slice();end[k]+=.15;line(origin,end,['#ff6666','#62db88','#779aff'][k],3);label(end,'XYZ'[k],['#ff6666','#62db88','#779aff'][k])}}
c.onpointerdown=e=>{drag=[e.clientX,e.clientY];c.setPointerCapture(e.pointerId)};c.onpointerup=()=>drag=null;c.onpointermove=e=>{if(!drag)return;yaw+=(e.clientX-drag[0])*.008;pitch+=(e.clientY-drag[1])*.008;drag=[e.clientX,e.clientY];draw()};c.onwheel=e=>{e.preventDefault();zoom=Math.max(.4,Math.min(4,zoom*Math.exp(-e.deltaY*.001)));draw()};document.getElementById('top').onclick=()=>{yaw=0;pitch=Math.PI/2;draw()};document.getElementById('side').onclick=()=>{yaw=Math.PI/2;pitch=0;draw()};document.getElementById('oblique').onclick=()=>{yaw=-.6;pitch=.5;zoom=2.3;draw()};onresize=draw;draw();</script></html>'''
(E/'index.html').write_text(html.replace('__ROWS__',rows).replace('__DATA__',json.dumps(d,ensure_ascii=False)),encoding='utf-8')
(E/'LAYOUT_AND_BUDGET.md').write_text((ROOT/'ff_prior_balanced_scene/docs/LAYOUT_AND_BUDGET.md').read_text(encoding='utf-8'),encoding='utf-8')
receipt=E/'preview_server.json'
if not receipt.exists():
    with socket.socket() as s:s.bind(('127.0.0.1',8770))
    with (E/'preview_stdout.log').open('ab') as out,(E/'preview_stderr.log').open('ab') as err:
        p=subprocess.Popen([sys.executable,'-m','http.server','8770','--bind','127.0.0.1','--directory',str(E)],
            stdin=subprocess.DEVNULL,stdout=out,stderr=err,cwd=ROOT,
            creationflags=subprocess.CREATE_NO_WINDOW|subprocess.CREATE_NEW_PROCESS_GROUP)
    receipt.write_text(json.dumps(dict(pid=p.pid,url='http://127.0.0.1:8770/index.html',started_at=time.time(),read_only=True,new_native_steps=0)),encoding='utf-8')
print(receipt.read_text())
