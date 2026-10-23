"""Check the new diagnosis/plan arithmetic and identities without physics."""
import json, hashlib, ast, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];PKG=ROOT/'ff_prior_momentum_diagnosis'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
d=read(PKG/'evidence/diagnosis.json');b=read(PKG/'evidence/budget_derivation.json')
s=read(PKG/'evidence/repair_spec.json');c=read(ROOT/'ff_prior_fullchain/control.json')
checks={}
checks['complete_saved_interval_count']=sum(p['states']-1 for p in d['phases'])==41750
checks['full_handoffs']=all(h['full836_equal'] and h['momentum_equal'] for h in d['handoffs'])
checks['source_preservation_receipts']=all(p['source_unchanged'] for p in d['phases'])
checks['momentum_reconstruction']=all(p['saved_momentum_reconstruction_max_abs']==0 for p in d['phases'])
checks['no_diagnostic_physics']=all(d[k]==0 for k in ('new_native_steps','new_replay_steps','new_coupled_calls','new_geometry_queries'))
checks['native_total']=sum(p['execution']+p['replay'] for p in b['phases'])==83500
checks['stage_cost']=4*83500==b['rk4_internal_force_stages']
checks['time_total']=sum(b['wall_caps_s'].values())==10800
checks['bytes_total']=sum(p['bytes'] for p in b['storage_envelope'])==b['storage_envelope_bytes']<=b['proposed_unit_caps']['bytes']
checks['source_input_proposal_hashes']=all(sha(ROOT/p)==h for p,h in s['proposal_source_input_hashes'].items())
checks['no_freeze_or_worker']=not s['production_implemented'] and not s['worker_started'] and not c['active_worker']
checks['closed_role_retained']=c['active_run_role']=='most_recent_closed_run'
checks['budget_extension_not_applied']=not b['ceilings_changed_now']
for p in (PKG/'tools').glob('*.py'):
    ast.parse(p.read_text(encoding='utf-8'));checks['syntax_'+p.name]=True
for p in (PKG/'evidence').glob('*.json'):read(p)
out=dict(status='PASS' if all(checks.values()) else 'FAIL',scope='metadata/arithmetic/AST/hash only; not physical qualification',checks=checks,new_native_steps=0,new_coupled_calls=0)
(PKG/'evidence/plan_checks.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
files=[dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in PKG.rglob('*') if p.is_file() and p.name!='file_manifest.json']
total=sum(p['bytes'] for p in files)
assert total<64*1024**2 and all(checks.values())
(PKG/'evidence/file_manifest.json').write_text(json.dumps(dict(files=files,total_bytes=total,local_npz_not_staged=True),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(status=out['status'],checks=len(checks),bytes=total,source_unchanged=True),ensure_ascii=False))
