"use strict";
const $=id=>document.getElementById(id), C={c:"#54dfc5",r:"#ffbd64",grid:"#26374a",muted:"#8298b1"};
let live=null,display=null,replay=null,replayTimer=null,denoiseTimer=null,connected=false,busyRequest=false,lastRun=null,denoiseManual=false;
let camera={yaw:-.55,pitch:.65,zoom:1,view:"perspective"},drag=null;
const statuses={idle:"准备就绪",running:"正在计算",paused:"计算已暂停",stopping:"正在终止",cancelled:"用户已终止",completed:"执行完成 · 等待核验结论",failed:"执行失败"};
function renderExperiment(){
  const c=live?.inference_config||{steps:60,candidates:5};
  $("inferenceStatus").textContent=`${c.steps} 步 DDIM；后 ${Math.ceil(c.steps*.3)} 步加入双臂参考、关节限值与碰撞梯度；${c.candidates} 条候选`;
  $("guidance").options[0].textContent=`开启 · 后 ${Math.ceil(c.steps*.3)} 步`;
  const stages=live?.experiment_stages||{},names={training:"全量训练",screen:"候选评分验证",refine:"启动与导数修正",validation:"全量验证",final:"新测试与对照"};
  const latest=Object.keys(names).filter(k=>stages[k]).at(-1);
  $("experimentStatus").textContent=latest?`${names[latest]}：${stages[latest].status}${stages[latest].name?" · "+stages[latest].name:""}`:"正在生成教师数据；150 条即开始中期训练。";
}
for(let j=0;j<17;j++){const o=document.createElement("option");o.value=j;o.textContent=j<10?`柔性 q${j+1}`:`刚性 q${j-9}`;$("joint").append(o);}
const fmt=(x,n=3)=>typeof x==="number"&&Number.isFinite(x)?x.toFixed(n):"—";
function el(tag,text,cls){const e=document.createElement(tag);if(text!==undefined)e.textContent=text;if(cls)e.className=cls;return e;}
function showError(e){$("alert").textContent=String(e.message||e);$("alert").hidden=false;}
async function api(path,body){const r=await fetch(path,body===undefined?{cache:"no-store"}:{method:"POST",headers:{"Content-Type":"application/json","X-Tracking-Client":"1"},body:JSON.stringify(body)});const d=await r.json();if(!r.ok)throw Error(d.error||`HTTP ${r.status}`);return d;}
async function command(path,body){if(busyRequest)return;busyRequest=true;$("alert").hidden=true;try{await api(path,body);await poll();}catch(e){showError(e);}finally{busyRequest=false;updateControls();}}
$("start").onclick=()=>{if(!$("seed").checkValidity()||$("seed").value===""||!$("samplingSeed").checkValidity()){showError("请输入合法的任务种子和模型采样种子。");return;}leaveReplay();command("/api/start",{mode:$("mode").value,seed:Number($("seed").value),duration:Number($("duration").value),speed:Number($("speed").value),sampling_seed:Number($("samplingSeed").value),guidance:$("guidance").value==="on",recording:$("recording").value});};
$("pause").onclick=()=>command("/api/control",{action:live?.status==="paused"?"resume":"pause"});
$("cancel").onclick=()=>command("/api/control",{action:"cancel"});
function updateControls(){const running=live&&["running","paused","stopping"].includes(live.status);$("start").disabled=!connected||running||busyRequest;$("pause").disabled=!connected||!running||live.status==="stopping"||busyRequest;$("cancel").disabled=$("pause").disabled;$("pause").textContent=live?.status==="paused"?"继续计算":"暂停计算";for(const id of ["mode","seed","duration","speed","samplingSeed","guidance","recording"])$(id).disabled=!!running;$("mode").querySelector('[value="diffusion"]').disabled=!live?.checkpoint_available;}
async function poll(){try{live=await api("/api/state");connected=true;$("connection").textContent="● 本地计算已连接";$("connection").className="pass";$("checkpoint").textContent=live.checkpoint_label;if(live.batch){$("batchProgress").textContent="教师生产："+live.batch.finished+" 次已执行（主要计划 "+live.batch.total+" 任务）；"+live.batch.passed+" 条通过教师筛选"+(live.batch.train_admitted!==undefined?"；可用训练 "+live.batch.train_admitted+"，目标 "+live.batch.training_target:"");}else{$("batchProgress").textContent="首轮数据：20 条完整教师轨迹";}if(live.training){$("dataStatus").textContent=live.training.train_tasks+" 个训练任务 / "+live.training.validation_tasks+" 个验证任务；按任务划分后切片，测试不入训练";$("trainingStatus").textContent="AdaLN Transformer · "+live.training.steps+" 步 · "+live.training.train_windows+" 个训练窗口";}const runs=live.runs||[];const selected=$("runs").value;if(Array.from($("runs").options).map(o=>o.value).join()!==runs.join()){$("runs").replaceChildren();if(!runs.length)$("runs").append(new Option("暂无记录",""));for(const r of runs)$("runs").append(new Option(r,r));if(runs.includes(selected))$("runs").value=selected;}if(!replay){display=live;render();}}catch(e){connected=false;$("connection").textContent="● 连接中断";$("connection").className="fail";}updateControls();}
async function polling(){await poll();renderExperiment();setTimeout(polling,350);}polling();
function render(){if(!display)return;const d=display,f=d.frame;$("source").textContent=replay?"历史回放":"实时数据";$("source").className="badge "+(replay?"warn":"");$("runId").textContent=d.run_id||"等待新任务";$("status").textContent=(replay?(replayTimer?"历史回放 · 正在播放":"历史回放 · 查看记录"):(statuses[d.status]||d.status))+` · ${{teacher:"教师",diffusion:"扩散", "feedback-only":"仅反馈"}[d.mode]||d.mode}`+(d.task_split?` · ${d.task_split}`:"")+(d.scene?` · 种子 ${d.scene.seed}`:"");document.querySelectorAll("[data-stage]").forEach(e=>e.classList.toggle("active",e.dataset.stage===d.stage));$("emptyScene").hidden=!!d.scene;
  if(lastRun!==d.run_id){lastRun=d.run_id;denoiseManual=false;$("denoiseStep").value=32;}
  $("simTime").textContent=`${fmt(f?.time||0,2)} / ${fmt(d.scene?.duration||Number($("duration").value),2)} s`;$("progress").max=d.scene?.duration||27;$("progress").value=f?.time||0;$("errorC").textContent=fmt(f?.error_c_mm);$("errorR").textContent=fmt(f?.error_r_mm);$("margin").textContent=fmt(f?.margin_mm);$("margin").className=f?.margin_mm<0?"fail":"";$("torque").textContent=f?`${fmt(f.torque_ratio*100,1)}%`:"—";$("physics").textContent=`${f?.physics_steps?.toLocaleString()||0} 次物理步进`;
  const logs=d.logs||[];$("logs").replaceChildren(...(logs.length?logs.map(x=>{const p=el("p");p.append(el("time",x.at),document.createTextNode(x.text));return p;}):[el("p",replay?"此处显示保存的事件，回放不会运行控制器。":"等待启动本地计算。") ]));
  renderResult();drawScene();drawErrors();drawDenoise();
}
function renderResult(){const r=display.result;$("resultBadge").className="badge";if(!r){$("resultBadge").textContent=display.failure?"失败":"待核验";$("result").replaceChildren(el("p",display.failure||"完成执行后显示全程误差、路径段误差、碰撞余量和计算延迟。",display.failure?"failure":"result-placeholder"));return;}
  const admissible=r.mode==="teacher"?r.accepted_for_teacher_dataset:(r.meets_stage_tracking&&r.meets_original_tracking_precision);$("resultBadge").textContent=r.failure?"失败 / 已终止":!r.full_task_executed?"短时试跑 · 不作性能结论":admissible?(r.mode==="teacher"?"教师样本通过":"精度指标通过 · 仍需独立评估"):"执行完成 · 指标未通过";$("resultBadge").className="badge "+(r.failure?"fail":admissible?"pass":"warn");const nodes=[];if(r.failure)nodes.push(el("div",r.failure,"failure"));const table=el("table",undefined,"result-table"),head=el("tr");for(const t of ["位置误差 / mm","全程 RMSE","全程最大","路径段 RMSE","最终"])head.append(el("th",t));const th=el("thead");th.append(head);table.append(th);const tb=el("tbody");for(const [key,label]of [["continuum","柔性臂"],["rigid","刚性臂"]]){const m=r.metrics[key],tr=el("tr");for(const t of [label,fmt(m.position_rmse_mm),fmt(m.position_max_mm),fmt(m.path_rmse_mm),fmt(m.final_position_mm)])tr.append(el("td",t));tb.append(tr);}table.append(tb);const wrap=el("div",undefined,"result-scroll");wrap.append(table);nodes.push(wrap);const facts=el("div",undefined,"result-facts");for(const t of [`全程最小余量 ${fmt(r.execution.minimum_pair_margin_m == null ? null : r.execution.minimum_pair_margin_m*1000)} mm`,`力矩饱和 ${r.execution.torque_saturation_count} 次`,`物理步进 ${r.execution.physics_steps}`,`采样+筛选 p95 ${fmt(r.sampling_and_selection_p95_ms,1)} ms`])facts.append(el("span",t));nodes.push(facts);$("result").replaceChildren(...nodes);
}
function canvas(id){const e=$(id),r=e.getBoundingClientRect(),scale=window.devicePixelRatio||1;if(e.width!==Math.round(r.width*scale)||e.height!==Math.round(r.height*scale)){e.width=Math.round(r.width*scale);e.height=Math.round(r.height*scale);}const ctx=e.getContext("2d");ctx.setTransform(scale,0,0,scale,0,0);ctx.clearRect(0,0,r.width,r.height);return {ctx,w:r.width,h:r.height};}
function line(ctx,points,color,width=1,dash=[]){if(!points.length)return;ctx.beginPath();points.forEach((p,i)=>i?ctx.lineTo(p[0],p[1]):ctx.moveTo(p[0],p[1]));ctx.strokeStyle=color;ctx.lineWidth=width;ctx.setLineDash(dash);ctx.stroke();ctx.setLineDash([]);}
function dot(ctx,p,color,r=3){ctx.beginPath();ctx.arc(p[0],p[1],r,0,Math.PI*2);ctx.fillStyle=color;ctx.fill();}
function drawScene(){const {ctx,w,h}=canvas("scene"),s=display?.scene;if(!s)return;const g=display.frame?.geometry||s.geometry;const all=[...g.continuum,...g.rigid,...s.reference_c,...s.reference_r,g.base,g.target];const min=[0,1,2].map(j=>Math.min(...all.map(p=>p[j]))),max=[0,1,2].map(j=>Math.max(...all.map(p=>p[j]))),center=min.map((x,j)=>(x+max[j])/2);const scale=Math.min(w/Math.max(max[0]-min[0],max[1]-min[1],1.4),h/1.6)*.74*camera.zoom;
  function proj(p){let [x,y,z]=p.map((v,j)=>v-center[j]);let a,b;if(camera.view==="xy"){a=x;b=y;}else if(camera.view==="xz"){a=x;b=z;}else if(camera.view==="yz"){a=y;b=z;}else{a=Math.cos(camera.yaw)*x-Math.sin(camera.yaw)*y;const d=Math.sin(camera.yaw)*x+Math.cos(camera.yaw)*y;b=Math.cos(camera.pitch)*z-Math.sin(camera.pitch)*d;}return[w/2+a*scale,h/2-b*scale];}
  for(let x=-.5;x<=3.5;x+=.25)line(ctx,[proj([x,-1.3,-.3]),proj([x,1.3,-.3])],"#23314480");for(let y=-1.25;y<=1.3;y+=.25)line(ctx,[proj([-.5,y,-.3]),proj([3.5,y,-.3])],"#23314480");
  function box(pos,rot,size,color){const pts=[];for(let i=0;i<8;i++){const local=size.map((v,j)=>((i>>j)&1?1:-1)*v/2);pts.push(proj(pos.map((v,j)=>v+local.reduce((sum,x,k)=>sum+rot[j][k]*x,0))));}for(let i=0;i<8;i++)for(let j=0;j<3;j++){const n=i^(1<<j);if(i<n)line(ctx,[pts[i],pts[n]],color,1.4);}}
  box(g.base,g.base_rotation,[.38,.5,.3],"#7293b5");box(g.target,g.target_rotation,[.28,.32,.28],"#9b9fd5");
  for(const o of s.obstacles){const p=proj(o.center);dot(ctx,p,"#ff798a22",Math.max(3,o.radius*scale));ctx.beginPath();ctx.arc(...p,Math.max(3,o.radius*scale),0,Math.PI*2);ctx.strokeStyle="#ff798aaa";ctx.lineWidth=1;ctx.stroke();}
  line(ctx,s.reference_c.map(proj),"#54dfc570",1.5,[5,5]);line(ctx,s.reference_r.map(proj),"#ffbd6470",1.5,[5,5]);for(const p of s.waypoints)dot(ctx,proj(p),"#54dfc590",2);
  const hist=display.history||[];line(ctx,hist.map(f=>proj(f.actual_c)),C.c,2);line(ctx,hist.map(f=>proj(f.actual_r)),C.r,2);line(ctx,g.continuum.map(proj),C.c,5);line(ctx,g.rigid.map(proj),C.r,5);g.rigid.forEach(p=>{dot(ctx,proj(p),"#192433",4);dot(ctx,proj(p),C.r,2);});dot(ctx,proj(g.continuum.at(-1)),"#edfff9",4);dot(ctx,proj(g.rigid.at(-1)),"#fff3df",4);
  if(display.frame){for(const [p,c]of [[display.frame.desired_c,C.c],[display.frame.desired_r,C.r]]){const [x,y]=proj(p);line(ctx,[[x-5,y],[x+5,y]],c);line(ctx,[[x,y-5],[x,y+5]],c);}}
  ctx.font="10px Segoe UI";for(const [p,t,c]of [[[.25,0,0],"X","#df8795"],[[0,.25,0],"Y",C.c],[[0,0,.25],"Z","#8faafa"]]){line(ctx,[proj([0,0,0]),proj(p)],c,1.5);const [x,y]=proj(p);ctx.fillStyle=c;ctx.fillText(t,x+4,y);}ctx.fillStyle="#9eafd0";const target=proj(g.target);ctx.fillText("漂浮目标",target[0]+12,target[1]-12);
}
function plotBase(id,xmax,ymin,ymax){const {ctx,w,h}=canvas(id),pad={l:45,r:14,t:16,b:25};const x=t=>pad.l+t/Math.max(xmax,1e-9)*(w-pad.l-pad.r),y=v=>h-pad.b-(v-ymin)/Math.max(ymax-ymin,1e-9)*(h-pad.t-pad.b);ctx.font="10px Consolas";for(let i=0;i<=4;i++){const v=ymin+(ymax-ymin)*i/4,py=y(v);line(ctx,[[pad.l,py],[w-pad.r,py]],C.grid);ctx.fillStyle=C.muted;ctx.textAlign="right";ctx.fillText(v.toFixed(Math.abs(ymax-ymin)<.1?3:2),pad.l-7,py+3);}for(let i=0;i<=4;i++){ctx.fillStyle=C.muted;ctx.textAlign="center";ctx.fillText((xmax*i/4).toFixed(1),x(xmax*i/4),h-6);}ctx.textAlign="left";return{ctx,x,y,w,h};}
function drawErrors(){const history=display?.error_history||display?.history||[],max=Math.max(.1,...history.flatMap(f=>[f.error_c_mm,f.error_r_mm]));const p=plotBase("errors",Math.max(1,history.at(-1)?.time||display?.frame?.time||0),0,max*1.15);for(const[key,color]of [["error_c_mm",C.c],["error_r_mm",C.r]])line(p.ctx,history.map(f=>[p.x(f.time),p.y(f[key])]),color,1.7);}
function drawDenoise(){
  const stage=$("candidateStage").value,stageData=display?.selection?.candidate_stages?.[stage];
  const steps=stage==="denoising"?(display?.denoising||[]):stageData?[{...stageData,step:1,total:1,time:display?.selection?.time??display?.frame?.time??0,phaseLabel:$("candidateStage").selectedOptions[0].textContent}]:[],joint=Number($("joint").value);
  $("animateDenoise").disabled=!steps.length;$("denoiseStep").disabled=!steps.length;
  $("denoiseStep").max=Math.max(0,steps.length-1);
  if(!denoiseManual)$("denoiseStep").value=Math.max(0,steps.length-1);
  const idx=Math.min(Number($("denoiseStep").value),steps.length-1),d=steps[idx];
  const curves=d?.joint_curves||[],horizon=d?.horizon||.5;
  const ys=curves.flatMap(c=>c.map(p=>p[joint])).filter(Number.isFinite);
  const min=ys.length?Math.min(...ys):-.1,max=ys.length?Math.max(...ys):.1;
  const margin=Math.max((max-min)*.15,.002),p=plotBase("denoise",horizon,min-margin,max+margin);
  const colors=[C.c,C.r,"#a6a1fa","#7bb8ef","#ee91ba"];
  const chosen=Math.abs((display?.selection?.time??-1)-(d?.time??-2))<1e-6?display?.selection?.selected_index:null;
  curves.forEach((c,k)=>line(p.ctx,c.map((q,i)=>[p.x(horizon*i/(c.length-1)),p.y(q[joint])]),colors[k%5],k===chosen?3.5:1.5,d.bounds_ok[k]?[]:[4,4]));
  if(d){
    $("denoiseLabel").textContent=`${d.step}/${d.total}`;
    const phase=d.phaseLabel||(d.guidance_active?"约束梯度正在注入":d.step===0?"高斯噪声初始化":"条件扩散去噪");
    $("denoiseInfo").textContent=`t = ${fmt(d.time,2)} s · ${phase}`;
    $("candidateInfo").textContent=`${d.bounds_ok.filter(Boolean).length} / ${curves.length} 条通过关节位置、速度、加速度边界。${chosen==null?"":`选中第 ${chosen+1} 条（粗线）。`}横轴：未来 ${fmt(horizon,1)} s；纵轴：rad。虚线为边界未通过。`;
  }else{
    $("denoiseLabel").textContent="—";
    $("denoiseInfo").textContent=display?.mode==="diffusion"?"等待模型采样…":"教师 / 仅反馈模式不执行扩散去噪";
    $("candidateInfo").textContent="扩散模式显示 5 条真实候选轨迹；可关闭生成时引导进行对照。";
    p.ctx.fillStyle=C.muted;p.ctx.textAlign="center";p.ctx.fillText("暂无扩散采样记录",p.w/2,p.h/2);
  }
}
$("joint").onchange=drawDenoise;$("denoiseStep").oninput=()=>{denoiseManual=true;stopDenoise();drawDenoise();};
function stopDenoise(){clearInterval(denoiseTimer);denoiseTimer=null;$("animateDenoise").textContent="播放去噪";}
$("animateDenoise").onclick=()=>{if(denoiseTimer){stopDenoise();return;}denoiseManual=true;$("denoiseStep").value=0;$("animateDenoise").textContent="暂停回看";drawDenoise();denoiseTimer=setInterval(()=>{const n=Number($("denoiseStep").value)+1;if(n>Number($("denoiseStep").max)){stopDenoise();return;}$("denoiseStep").value=n;drawDenoise();},160);};
document.querySelectorAll("[data-view]").forEach(b=>b.onclick=()=>{camera.view=b.dataset.view;document.querySelectorAll("[data-view]").forEach(v=>v.classList.toggle("active",v===b));drawScene();});$("resetView").onclick=()=>{camera={yaw:-.55,pitch:.65,zoom:1,view:"perspective"};document.querySelector('[data-view="perspective"]').click();};
$("scene").onpointerdown=e=>{drag={x:e.clientX,y:e.clientY};$("scene").setPointerCapture(e.pointerId);};$("scene").onpointermove=e=>{if(!drag)return;camera.view="perspective";camera.yaw+=(e.clientX-drag.x)*.006;camera.pitch=Math.max(-1.4,Math.min(1.4,camera.pitch+(e.clientY-drag.y)*.006));drag={x:e.clientX,y:e.clientY};drawScene();};$("scene").onpointerup=()=>drag=null;$("scene").onpointercancel=()=>drag=null;$("scene").addEventListener("wheel",e=>{e.preventDefault();camera.zoom=Math.max(.5,Math.min(3,camera.zoom*Math.exp(-e.deltaY*.001)));drawScene();},{passive:false});
new ResizeObserver(()=>{drawScene();drawErrors();drawDenoise();}).observe($("scene"));
function stopReplay(){clearInterval(replayTimer);replayTimer=null;$("playReplay").textContent="播放回放";}
function leaveReplay(){stopReplay();stopDenoise();replay=null;display=live;$("replayControls").hidden=true;render();}
$("returnLive").onclick=leaveReplay;
$("loadReplay").onclick=async()=>{if(!$("runs").value){showError("尚无可回放的运行。");return;}try{const data=await api("/api/replay/"+encodeURIComponent($("runs").value));stopReplay();stopDenoise();replay={...data,frames:data.events.filter(e=>e.kind==="frame").map(e=>e.value)};if(!replay.frames.length)throw Error("此记录没有仿真帧；请查看对应运行的失败文件。");$("replaySlider").max=replay.frames.length-1;$("replaySlider").value=replay.frames.length-1;$("replayControls").hidden=false;showReplay();}catch(e){replay=null;showError(e);}};
function showReplay(){if(!replay)return;const index=Number($("replaySlider").value),f=replay.frames[index],atEnd=index===replay.frames.length-1;const scene=replay.events.find(e=>e.kind==="scene")?.value,run=replay.events.find(e=>e.kind==="run")?.value,result=replay.events.find(e=>e.kind==="result")?.value;let denoising=[],selection=null;for(const e of replay.events){if(e.kind==="selection"&&(e.value.time??-1)<=f.time+1e-6)selection=e.value;if(e.kind==="denoising"&&e.value.time<=f.time+1e-6){if(e.value.step===0)denoising=[];denoising.push(e.value);}}display={run_id:replay.run_id,scene,frame:f,mode:run?.mode,status:atEnd?(result?.completed?"completed":"failed"):"paused",stage:atEnd?"verification":"execution",history:replay.frames.slice(0,index+1),error_history:(replay.error_history||[]).filter(x=>x.time<=f.time+1e-6),denoising,selection,result:atEnd?result:null,failure:atEnd?result?.failure:null,logs:[]};$("replayTime").textContent=fmt(f.time,2)+" s";render();}
$("replaySlider").oninput=()=>{stopReplay();showReplay();};$("playReplay").onclick=()=>{if(replayTimer){stopReplay();return;}if(Number($("replaySlider").value)>=Number($("replaySlider").max))$("replaySlider").value=0;$("playReplay").textContent="暂停回放";showReplay();replayTimer=setInterval(()=>{let i=Number($("replaySlider").value)+1;if(i>Number($("replaySlider").max)){stopReplay();return;}$("replaySlider").value=i;showReplay();},100);};

$("candidateStage").onchange=()=>{denoiseManual=false;drawDenoise();};
