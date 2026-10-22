"""Target only the two sub-nanoradian reconstruction discrepancies.
No physical or optimizer calls. The nominal is one saved 17x17 damped linear system.
"""
import hashlib,json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits,threadpool_info
ROOT=Path(__file__).resolve().parents[4];HERE=Path(__file__).resolve().parent
OUT=ROOT/'artifacts/v7_1/contact_steer_001'
rows=[]
for i in [42,46]:
    with np.load(OUT/'units/u02/tree/linear'/f'{i:03d}'/'jacobian.npz') as a:
        J=a['hold_j'];e=a['hold_e'];nom=a['nominal'];H=J.T@J+.001*np.eye(17);b=J.T@e
        row={'iteration':i,'damped_matrix_condition_number':float(np.linalg.cond(H)),'threadpool_before':threadpool_info()}
        for threads in [None,2,1]:
            with threadpool_limits(limits=threads):
                x=-np.linalg.solve(J.T@J+.001*np.eye(17),J.T@e);x*=min(1.,.12/max(float(np.max(np.abs(x))),1e-12))
                row[str(threads)]={'maximum_absolute_nominal_error_rad':float(np.max(np.abs(x-nom))),'normalized_system_residual_of_reconstructed':float(np.linalg.norm(H@x+b)/(np.linalg.norm(H)*np.linalg.norm(x)+np.linalg.norm(b))),'exact':bool(np.array_equal(x,nom))}
        row['max_difference_acceptable_at_1e_8_rad']=row['None']['maximum_absolute_nominal_error_rad']<=1e-8
        rows.append(row)
result={'scope':'targeted damped linear arithmetic only, no new trajectory or optimizer evaluation','rows':rows,'all_match_original_two_threads_exactly':all(r['2']['exact'] for r in rows)}
(HERE/'nominal_arithmetic_diagnostic.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result,indent=2))
