from pathlib import Path
import json,numpy as np
ROOT=Path(__file__).resolve().parents[4];OUT=Path(__file__).resolve().parent;RUN=ROOT/'artifacts/v7_1/dual_obstacle_chain_001'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
# Correct the summary's original overly strict combined name without repeating neural work.
n=read(OUT/'proposal_budget_evidence.json');n.pop('neural_exact',None)
n['network_raw_bit_exact']=all(r['network_calls']==16 and r['raw_max_error']==0 for r in n['neural'])
n['public_bow_proposal_atol_1e12']=all(r['proposal_max_error']<=1e-12 for r in n['neural'])
n['public_bow_note']='Public bow is a fresh small linear solve; this review observes up to 1.01e-13 rad arithmetic discrepancy and does not label it bit exact.'
(OUT/'proposal_budget_evidence.json').write_text(json.dumps(n,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
a=np.load(RUN/'dynamic/execution/native_trace.npz');c=read(RUN/'dynamic/contract.json');points=np.array(c['waypoints']['waypoint_points_m']);centers=(points[:-1]+points[1:])/2;bounds=np.r_[4.5,4.5+np.cumsum(c['waypoints']['segment_durations_s'])];mid=(bounds[:-1]+bounds[1:])/2;local=a['time']-a['time'][0]
rows=[]
for i,t in enumerate(mid):
 k=int(np.argmin(abs(local-t)));p=a['ee'][k,0,:3,3];rows.append(dict(segment=i+1,center_m=centers[i].tolist(),scheduled_midpoint_local_s=float(t),nearest_sample_index=k,nearest_sample_local_s=float(local[k]),time_offset_ms=float((local[k]-t)*1000),saved_actual_tip_position_m=p.tolist(),saved_tip_to_center_m=float(np.linalg.norm(p-centers[i])),saved_tip_minus_sphere_radius_mm=float((np.linalg.norm(p-centers[i])-.012)*1000)))
out=dict(scope='Algebra on already-saved tip coordinates and frozen centers only. This is not whole-body collision clearance or a new geometry certificate.',segment_midpoints=rows)
(OUT/'saved_coordinate_evidence.json').write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
print(json.dumps(out,indent=2))
