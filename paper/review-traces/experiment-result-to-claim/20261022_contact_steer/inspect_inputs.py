"""Read saved evidence structure only. No repository imports or physical calls."""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT/'artifacts/v7_1/contact_steer_001'
def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
run=read(OUT/'run.json'); ident=run['identity']; contract=read(OUT/'physical_contract.json')
print('run keys',list(run),'identity keys',list(ident))
print('source count',len(ident['source_hashes']),'bound count',len(ident['binding']['files']))
print('protocol',json.dumps(ident['protocol']))
print('contract keys',list(contract),'physical identity keys',list(contract['identity']))
print('task',json.dumps(ident['selection']['entries'][0]))
print('ledger',json.dumps(read(OUT/'ledger.json')))
for u in ident['selection']['units']:
 d=OUT/'units'/u['id']; a=read(d/'tree/attempts.json')
 print('UNIT',u['id'],'result',json.dumps(read(d/'tree/result.json')))
 print('sample attempt',json.dumps(a[0]))
 for p in [d/'tree/nodes.npz',d/'tree/linear/000/jacobian.npz',d/'tree/edges/0000/trace.npz']:
  with np.load(p) as z: print('array shapes',p.relative_to(ROOT).as_posix(),{k:list(z[k].shape) for k in z.files})
 print('linear meta',json.dumps(read(d/'tree/linear/000/jacobian.json')))
 print('edge report',json.dumps(read(d/'tree/edges/0000/report.json')))
