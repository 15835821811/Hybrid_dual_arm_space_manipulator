"""Display geometry derived solely from saved EE and goal SE(3) matrices."""
import numpy as np
import mujoco

AXIS_COLORS=np.asarray([[1.,.15,.15,1.],[.15,.9,.2,1.],[.2,.4,1.,1.]],dtype=np.float32)
PATH_COLORS=np.asarray([[.15,.9,1.,.85],[1.,.65,.15,.85]],dtype=np.float32)


def frame_segments(pose,length=.075):
    pose=np.asarray(pose)
    return [(pose[:3,3].copy(),pose[:3,3]+length*pose[:3,j]) for j in range(3)]


def connector(scene,a,b,color,width=.002,arrow=False,label=''):
    if scene.ngeom>=scene.maxgeom:raise RuntimeError('VISUAL_GEOMETRY_CAPACITY')
    g=scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_ARROW if arrow else mujoco.mjtGeom.mjGEOM_CAPSULE,
        np.zeros(3),np.zeros(3),np.eye(3).ravel(),np.asarray(color,dtype=np.float32))
    mujoco.mjv_connector(g,mujoco.mjtGeom.mjGEOM_ARROW if arrow else mujoco.mjtGeom.mjGEOM_CAPSULE,
        width,np.asarray(a,dtype=float),np.asarray(b,dtype=float))
    if label:g.label=label
    scene.ngeom+=1


def add_saved_overlays(scene,ee,goals,index,trajectory_indices):
    for arm in range(2):
        for target,pose,label in ((False,ee[index,arm],f'EE{arm}'),(True,goals[arm],f'Goal{arm}')):
            for axis,(a,b) in enumerate(frame_segments(pose,.09 if target else .065)):
                connector(scene,a,b,AXIS_COLORS[axis],.003 if target else .002,True,label if axis==2 else '')
        ids=np.asarray(trajectory_indices);ids=ids[ids<=index]
        if not len(ids) or ids[-1]!=index:ids=np.r_[ids,index]
        for a,b in zip(ids[:-1],ids[1:]):
            p,q=ee[int(a),arm,:3,3],ee[int(b),arm,:3,3]
            if np.linalg.norm(p-q)>1e-10:connector(scene,p,q,PATH_COLORS[arm],.0015)


JS_OVERLAYS='''
function segment(a,b,color,width=2,dash=false){const p=project(a),q=project(b);ctx.strokeStyle=color;ctx.lineWidth=width*devicePixelRatio;ctx.setLineDash(dash?[6*devicePixelRatio,4*devicePixelRatio]:[]);ctx.beginPath();ctx.moveTo(...p);ctx.lineTo(...q);ctx.stroke();ctx.setLineDash([])}
function frame(T,label,target){let origin=T.slice(0,3).map(r=>r[3]);let colors=['#ff4545','#39df5e','#518aff'];for(let j=0;j<3;j++){let end=origin.map((v,k)=>v+(target ? .09 : .065)*T[k][j]);segment(origin,end,colors[j],target?3:2,target);let q=project(end);ctx.fillStyle=colors[j];ctx.font=(11*devicePixelRatio)+'px system-ui';ctx.fillText(label+' '+['X','Y','Z'][j],q[0]+3,q[1]-3)}}
function savedOverlays(){for(let arm=0;arm<2;arm++){for(let k=1;k<D.ee.length;k++)segment(D.ee[k-1][arm],D.ee[k][arm],arm?'#e6a24335':'#37d9ee35',1);for(let k=1;k<=i;k++)segment(D.ee[k-1][arm],D.ee[k][arm],arm?'#ffbd53':'#40dfef',3);frame(D.ee_poses[i][arm],'EE'+arm,false);frame(D.goal_poses[i][arm],'Goal'+arm,true)}}
'''
