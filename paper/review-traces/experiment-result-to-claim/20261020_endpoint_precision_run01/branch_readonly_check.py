import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[4];DEST=Path(__file__).parent
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
index={e['sample_id']:e for e in read(ROOT/'artifacts/v7_1/expand256_train_dev_001/training_input_audit.json')['records']}
result=[]
for path in sorted((ROOT/'artifacts/v7_1/endpoint_precision_bounded_001/diagnostic').glob('*/gradient_0.json')):
    p=read(path);goals=np.array(read(ROOT/index[p['sample_id']]['record'])['task']['condition']['goals'])
    a=np.load(path.with_suffix('.npz'))
    def q2(ee):
        ee=np.asarray(ee);pos=np.linalg.norm(ee[:,:3,3]-goals[:,:3,3],axis=1)/.005
        rot=Rotation.from_matrix(goals[:,:3,:3].transpose(0,2,1)@ee[:,:3,:3]).magnitude()/np.deg2rad(1)
        return np.r_[pos,rot]**2
    central=q2(p['central']['ee']);active=int(np.argmax(central))
    pert=np.array([[q2(e) for e in pair] for pair in a['perturbation_ee']])
    fixed=pert[:,:,active]+.05*pert.mean(axis=2)
    g=(fixed[:,1]-fixed[:,0])/(2*p['epsilon'])
    direction=np.array(p['direction']);pred=float(g@direction);obs=p['direction_slope']
    branches=np.argmax(pert,axis=2)
    half_errors=[]
    for h in p['half_checks']:
        ls=[q2(s['ee']).max()+.05*q2(s['ee']).mean() for s in h['samples']]
        slope=(ls[1]-ls[0])/p['epsilon'];err=abs(slope-a['gradient'][h['dimension']])/max(abs(slope),abs(a['gradient'][h['dimension']]),1e-6)
        assert abs(err-h['relative_error'])<1e-10
        half_errors.append(err)
    direction_losses=[q2(s['ee']).max()+.05*q2(s['ee']).mean() for s in p['direction_samples']]
    measured=(direction_losses[1]-direction_losses[0])/(2*p['epsilon'])
    assert abs(measured-obs)<1e-9
    direction_err=abs(measured-np.dot(a['gradient'],direction))/max(abs(measured),abs(np.dot(a['gradient'],direction)),1e-6)
    assert abs(direction_err-p['direction_relative_error'])<1e-10
    result.append({'case':path.parent.name,'status':p['status'],'central_active_group':active,
       'coordinate_pairs_switching_max':int(np.count_nonzero(branches[:,0]!=branches[:,1])),
       'direction_probe_active_groups':[int(np.argmax(q2(s['ee']))) for s in p['direction_samples']],
       'direction_relative_error':direction_err,'half_relative_errors':half_errors,
       'fixed_central_branch_direction_error':abs(pred-obs)/max(abs(pred),abs(obs),1e-6),
       'original_gradient_not_changed':True,'no_new_physics':True})
(DEST/'branch_checks.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps([r for r in result if r['status']!='PASS'],indent=2))
