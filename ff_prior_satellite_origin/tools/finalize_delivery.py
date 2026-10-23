"""Read-only media presentation corrections; never import a simulation module."""
from pathlib import Path
import hashlib, json, re, subprocess, time

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'delivery_001'
EVIDENCE = ROOT / 'evidence'

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    started = time.time()
    page = (DEST/'index.html').read_text(encoding='utf-8')
    # Remove duplicated figure sections in the original media builder.
    seen = set()
    def dedup(match):
        key = match.group(1)
        if key in seen:
            return ''
        seen.add(key)
        return match.group(0)
    page = re.sub(r'<section><h2>(\d\d_[^<]+)</h2>.*?</section>', dedup, page, flags=re.S)
    page = page.replace('失败只展示实际前缀。', '本次实际执行到83.5秒，整体验收因机器人动量门失败。')
    note = '<section class="badge"><h2>验收结果：28/29项通过，机器人动量门失败</h2><p>卫星由2秒公开推力起动，随后关机追踪27秒；完整重放14,501帧×836项差为0。54.5秒继承状态已超过按本段交换量计算的原动量阈值。原门未修改，不能将本次记录称为全门合格。</p><p>EE0/Goal0为连续体，EE1/Goal1为刚性臂；实线为EE轴，虚线为Goal轴，RGB对应XYZ。高精度跟踪时两套轴接近重合，可旋转、缩放查看。W1–W7属于已完成的前驱七航点，本段连续体保持W7。</p></section>'
    page = page.replace('<section><h2>交互三维', note+'<section><h2>交互三维', 1)
    page = page.replace('<button id="play">播放/暂停</button>', '<button id="play">播放/暂停</button> <button id="burn">54.5秒起动</button> <button id="coast">56.5秒关机</button> <button id="end">83.5秒末态</button>')
    label = '''let labelBoxes=[];
function label(p,text,color='#eaf3ff'){let a=project(p),d=devicePixelRatio;ctx.font=(11*d)+'px system-ui';let w=ctx.measureText(text).width+6*d,h=14*d,x=a[0]+4*d,y=a[1]-4*d;for(let n=0;n<80;n++){x=Math.max(2*d,Math.min(c.width-w-2*d,a[0]+4*d+(n>35?90*d:0)));y=Math.max(h,Math.min(c.height-3*d,a[1]-4*d+(n%2?1:-1)*Math.ceil(n/2)*h));if(!labelBoxes.some(b=>x<b[0]+b[2]&&x+w>b[0]&&y-h<b[1]&&y>b[1]-h))break;}labelBoxes.push([x,y,w]);if(Math.abs(y-a[1])>16*d||Math.abs(x-a[0])>20*d){ctx.strokeStyle=color+'88';ctx.lineWidth=d;ctx.beginPath();ctx.moveTo(...a);ctx.lineTo(x,y-h/2);ctx.stroke();}ctx.fillStyle='#101923cc';ctx.fillRect(x-2*d,y-h,w,h+2*d);ctx.fillStyle=color;ctx.fillText(text,x,y-2*d);}
'''
    page = re.sub(r'function label\(.*?\nfunction dot', label+'function dot', page, count=1, flags=re.S)
    page = page.replace('function draw(){c.width=', 'function draw(){labelBoxes=[];c.width=', 1)
    page = re.sub(r"else rows='<tr><td colspan=\"3\">.*?</td></tr>';", "else rows='<tr><td colspan=\"3\">No synchronous body diagnostic; no distance interpolation</td></tr>';", page)
    page = re.sub(r"document.getElementById\('info'\).textContent=`.*?`;", "document.getElementById('info').textContent=`t=${D.time[i].toFixed(3)} s | EE position error: ${D.position_mm[i].map(x=>x.toFixed(3)).join(' / ')} mm | whole-body gap: ${D.clearance_mm[i]===null?'NA':D.clearance_mm[i].toFixed(3)} mm`;", page, count=1)
    jumps = "for(const [name,t] of [['burn',54.5],['coast',56.5],['end',83.5]])document.getElementById(name).onclick=()=>{let j=D.time.findIndex(x=>x>=t-1e-8);i=j<0?D.time.length-1:j;playing=false;playbackAnchor=null;draw()};"
    page = page.replace('draw();requestAnimationFrame(tick);</script>', jumps+'draw();requestAnimationFrame(tick);</script>')
    (DEST/'index.html').write_text(page, encoding='utf-8')
    video_updates=[]
    for name in ('overview','top'):
        tmp=DEST/(name+'_labelled.mp4')
        filt="drawbox=x=0:y=ih-36:w=iw:h=36:color=black@0.85:t=fill,drawtext=fontfile='C\\:/Windows/Fonts/arial.ttf':text='APP_DEV | new segment 54.5-83.5 s FAILED - robot momentum gate':fontcolor=0xffcd78:fontsize=18:x=12:y=h-28"
        subprocess.run(['ffmpeg','-v','error','-y','-i',str(DEST/(name+'.mp4')),'-vf',filt,'-c:v','libx264','-threads','2','-crf','20','-preset','fast','-an',str(tmp)],check=True)
        tmp.replace(DEST/(name+'.mp4'))
        video_updates.append(name+'.mp4')
    manifest=json.loads((DEST/'manifest.json').read_text(encoding='utf-8'))
    manifest['presentation_updates']=['deduplicated figures','momentum failure explanation','nonoverlapping interactive labels with leader lines','burn/coast/end jump buttons','permanent failed-new-segment video caption']
    manifest['presentation_wall_s']=time.time()-started
    manifest['files']={p.name:{'bytes':p.stat().st_size,'sha256':digest(p)} for p in DEST.iterdir() if p.is_file() and p.name!='manifest.json'}
    manifest['output_bytes']=sum(v['bytes'] for v in manifest['files'].values())
    (DEST/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'videos':video_updates,'wall_s':manifest['presentation_wall_s'],'output_bytes':manifest['output_bytes'],'new_physics_steps':0}))

if __name__=='__main__':
    main()
