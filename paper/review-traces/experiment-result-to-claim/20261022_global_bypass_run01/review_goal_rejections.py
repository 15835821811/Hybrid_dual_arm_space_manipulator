"""Classify every saved failed goal line trial; read-only, no physics."""
from pathlib import Path
import json,numpy as np,collections,hashlib
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[3];RUN=ROOT/'artifacts/v7_1/global_bypass_001'
def js(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def dg(x):return hashlib.sha256(json.dumps(x.tolist(),sort_keys=True,separators=(',',':')).encode()).hexdigest()
counts=collections.Counter();rows=[]
for td in RUN.glob('units/*/candidates/*/tree'):
    n=np.load(td/'nodes.npz');edges=[js(p) for p in (td/'edges').glob('*/report.json')]
    for hp in (td/'goal_connections').glob('*/iterations.json'):
        par=int(hp.parent.name);q=n['q'][par].copy();hsh=dg(n['state'][par])
        for it,rec in enumerate(js(hp)):
            a=np.load(hp.parent/f'iteration_{it:02d}.npz');cur=q.copy()
            for trial in rec['trials']:
                target=cur+trial['scale']*a['delta']
                matches=[e for e in edges if e['parent_state_hash']==hsh and np.max(abs(np.asarray(e['target'])-target))<1e-12 and abs(e['duration']-(27-n['elapsed'][par]))<1e-10]
                if not trial['edge_pass']:
                    if not matches:kind='NO_TRACE'
                    else:
                        e=max(matches,key=lambda e:e['edge_id'])
                        kind='COLLISION_EARLY_STOP' if not e['complete'] and e['minimum_clearance_m']<0 else 'NUMERIC_FAIL' if not e['numerical_ok'] else 'MAX_DENSITY_STEP_FAIL' if e['intervals']==512 and e['max_geometry_step_m']>.002 else 'OTHER'
                    counts[kind]+=1
                    rows.append({'candidate':str(td.parent.relative_to(RUN)),'parent':par,'iteration':it,'scale':trial['scale'],'classification':kind,'edge_ids':[e['edge_id'] for e in matches]})
                elif trial.get('accepted'):q=target
result={'counts':dict(counts),'rows':rows,'method':'saved parent state hash and reconstructed target/duration matching; no physics'}
(OUT/'goal_rejection_classification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(dict(counts))
