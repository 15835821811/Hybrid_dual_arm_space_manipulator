// Offline presentation QA only. No browser automation, physics or scientific re-audit.
const fs=require('fs'),path=require('path'),vm=require('vm'),cp=require('child_process');
const pkg=path.resolve(__dirname,'..'),dest=path.join(pkg,'delivery_002');
const html=fs.readFileSync(path.join(dest,'index.html'),'utf8');
const refs=[...html.matchAll(/(?:src|href)="([^"#]+)"/g)].map(x=>x[1]);
const missing=refs.filter(p=>!p.includes('://')&&!fs.existsSync(path.join(dest,p)));
if(missing.length)throw new Error('Missing media '+missing.join(','));
const counts={};
const context=id=>new Proxy({measureText:t=>({width:t.length*6})},{get:(o,k)=>k in o?o[k]:(...args)=>{counts[id]=(counts[id]||0)+1;}});
const elements={};
const get=id=>elements[id]??=(['c','trail0','trail1'].includes(id)?{clientWidth:id==='c'?1100:550,getContext:()=>context(id),setPointerCapture:()=>{}}:{});
const sandbox={document:{getElementById:get},devicePixelRatio:1,window:{},requestAnimationFrame:()=>{}};
vm.createContext(sandbox);
vm.runInContext(html.match(/<script>([\s\S]*?)<\/script>/)[1],sandbox,{timeout:20000});
const outcome=vm.runInContext(`(()=>{let invalid=[];for(let j=0;j<D.time.length;j++){i=j;draw();if(!Number.isFinite(D.time[i])||D.ee_poses[i].length!==2||D.goal_poses[i].length!==2)invalid.push(j);}let jumps={};for(const key of ['burn','coast','end']){document.getElementById(key).onclick();jumps[key]=D.time[i];}return{frames:D.time.length,invalid,first_time:D.time[0],last_time:D.time.at(-1),jumps,waypoints:D.waypoints.length,execution_level:D.execution_level}})()`,sandbox,{timeout:30000});
if(outcome.invalid.length||outcome.waypoints!==7||!counts.trail0||!counts.trail1)throw new Error('Invalid display or unsynchronised trails');
const videos=['overview','top'].map(v=>{
  const data=JSON.parse(cp.execFileSync('ffprobe',['-v','error','-select_streams','v:0','-count_frames','-show_entries','stream=nb_read_frames,duration,width,height','-of','json',path.join(dest,v+'.mp4')],{encoding:'utf8',windowsHide:true}));
  const s=data.streams[0];if(+s.nb_read_frames!==outcome.frames||Math.abs(+s.duration-83.5)>1e-6)throw new Error('Video mismatch '+v);
  return {file:v+'.mp4',...s};
});
const figures=[...html.matchAll(/<h2>(\d\d_[^<]+)<\/h2>/g)].map(x=>x[1]);
if(new Set(figures).size!==figures.length)throw new Error('Duplicate figure sections');
const csv=fs.readFileSync(path.join(dest,'full_actual_and_target_ee_xyz.csv'),'utf8').trim().split(/\r?\n/);
if(csv.length!==41752||csv[0].split(',').length!==13)throw new Error('CSV shape mismatch');
const result={status:'PASS',scope:'offline resource and JavaScript execution with a canvas/DOM test double; not browser visual verification or physical validation',resource_references:refs.length,missing_resources:missing,...outcome,canvas_draw_calls:counts,videos,unique_figure_sections:figures.length,exported_actual_states:csv.length-1,visual_review:['overview near first red highlight','top final video frame','EE 3D plot with separated phase caption'],new_native_steps:0,new_coupled_calls:0,new_geometry_queries:0,new_resume:false};
fs.writeFileSync(path.join(pkg,'evidence','trajectory_update_qa.json'),JSON.stringify(result,null,2)+'\n');
console.log(JSON.stringify(result));
