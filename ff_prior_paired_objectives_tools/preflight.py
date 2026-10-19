"""Read-only source/data check; no new training or physics experiments."""
from pathlib import Path
import sys,time,xml.etree.ElementTree as ET
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ff_prior.storage import digest,file_digest
from ff_prior_coverage_train.storage import atomic_json
from ff_prior_paired_objectives.protocol import ROOT,SOURCE
from ff_prior_paired_objectives.identity import identity_for,sources

if __name__=='__main__':
    started=time.time();identity,context=identity_for(SOURCE)
    assert all(file_digest(ROOT/p)==h for p,h in identity['source_files'].items())
    path=ROOT/'ff_prior_paired_objectives/evidence'
    r={'status':'PASS','source_files':len(identity['source_files']),'source_modules':len(sources()),
       'identity_hash':digest(identity),'source_hashes':sources(),'comparison_hash':identity['comparison_entries_hash'],
       'wall_s':time.time()-started,'TEST_read':False,'new_training_or_physics_calls':0}
    atomic_json(path/'preflight.json',r)
    print({k:v for k,v in r.items() if k!='source_hashes'},flush=True)
