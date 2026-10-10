"""Compact C3 hypothesis evidence and test coverage; zero model or physics calls."""
import csv
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from v6_4.route_optimizer_protocol import read,write,sha


def main():
    import numpy as np
    out=ROOT/'v6_4/c4a_evidence/review_01';out.mkdir(parents=True,exist_ok=False)
    snap=ROOT/'v6_4/releases/search_aware_warmstart_20261008_01/snapshot'
    tasks=sorted((snap/'test_search').iterdir());cases=[]
    for taskdir in tasks:
        tid=taskdir.name;paths={m:taskdir/m/'planning'/tid/'candidate_registry.json' for m in ('R','D')}
        r,d=[read(paths[m]) for m in ('R','D')]
        assert all(r[i]['plan_sha256']==d[i]['plan_sha256'] and r[i]['prediction_metrics']==d[i]['prediction_metrics'] for i in (0,2))
        seeds=[]
        for method,rows in [('R',r),('D',d)]:
            for x in rows[:4]:
                z=np.asarray(x.get('raw_z_m',(x.get('plan') or {}).get('z_m')),dtype=float).reshape(6,2)
                diag=x.get('raw_seed_diagnostics') or {}
                seeds.append(dict(method=method,candidate_id=x['candidate_id'],family=x['family'],source=x['source'],
                    z_m=z.tolist(),interval_norms_mm=(np.linalg.norm(z,axis=1)*1000).tolist(),
                    inactive_zero=bool(np.all(z[[0,3,4,5]]==0)),raw_finite=bool(np.isfinite(z).all()),
                    interval_norm_legal=bool(np.all(np.linalg.norm(z,axis=1)<=.020)),
                    raw_rejection=diag.get('rejection_reason'),nominal_passed=x['prediction_admissible'],
                    I_support=(x.get('prediction_metrics') or {}).get('I_support'),
                    L_full=(x.get('prediction_metrics') or {}).get('L_full'),
                    d_support=(x.get('prediction_metrics') or {}).get('d_support')))
        cases.append(dict(task_id=tid,common_rule_seeds_exactly_equal=True,seeds=seeds,
            D8_polls=[{k:x.get(k) for k in ('candidate_id','coordinate','sign','parent_candidate_id','preference_center','origin_source','x_m')} for x in d[4:8]],
            D8_B30_count=sum(x.get('prediction_admissible',False) and (x.get('prediction_metrics') or {}).get('d_support',-1)>=.03 for x in d[:8]),
            R_C03_B30=r[3]['prediction_metrics']['d_support']>=.03,
            source_sha256={m:sha(paths[m]) for m in paths}))
    write(out/'C3_B_mechanism.json',cases)
    unique=set();suites={};failures=[]
    for p in sorted((ROOT/'docs/audit_receipts/c4a').glob('*/junit.xml')):
        tests=ET.parse(p).findall('.//testcase');passed=0
        for c in tests:
            key=c.attrib.get('classname','')+'::'+c.attrib.get('name','')
            if c.find('failure') is not None or c.find('error') is not None:
                failures.append(dict(suite=p.parent.name,test=key))
            elif c.find('skipped') is None: unique.add(key);passed+=1
        suites[p.parent.name]=dict(cases=len(tests),passed=passed,junit_sha256=sha(p))
    write(out/'test_coverage.json',dict(unique_passed=len(unique),test_ids=sorted(unique),suites=suites,retained_failures=failures,
        fix='namespace-package collection uses pytest --import-mode=importlib; no old source edit',
        tests_are_not_physical_performance=True))
    print(json.dumps(dict(c3_tasks=len(cases),unique_tests_passed=len(unique),retained_failures=failures)))


if __name__=='__main__':main()
