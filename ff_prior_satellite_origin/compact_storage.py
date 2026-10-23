"""Lossless compact JSON for dense events; identical numeric precision."""
import os,json
from pathlib import Path
from ff_prior.storage import plain
from ff_prior_pilot_train.storage import retry_write
def json_size(value):
    return len(json.dumps(value,default=plain,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8'))
def _write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    with tmp.open('w',encoding='utf-8') as f:
        json.dump(value,f,default=plain,sort_keys=True,separators=(',',':'),allow_nan=False)
        f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
def atomic_json(path,value):return retry_write(_write,path,value)
