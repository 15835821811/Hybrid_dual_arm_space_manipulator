"""Start only a loopback read-only saved-media server; never runs simulation."""
import json,socket,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
E=ROOT/'ff_prior_horizon_restore/evidence'
receipt=E/'visual_server.json'
assert not receipt.exists(),'Inspect existing media server instead of duplicating'
with socket.socket() as s:s.bind(('127.0.0.1',8769))
directory=ROOT/'artifacts/v7_1/horizon_restore_001/units/u00/visuals'
with (E/'media_server.stdout.log').open('ab') as stdout,(E/'media_server.stderr.log').open('ab') as stderr:
    p=subprocess.Popen([sys.executable,'-m','http.server','8769','--bind','127.0.0.1','--directory',str(directory)],
        cwd=ROOT,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,
        creationflags=subprocess.CREATE_NO_WINDOW|subprocess.CREATE_NEW_PROCESS_GROUP)
v={'url':'http://127.0.0.1:8769/index.html','pid':p.pid,'started_at':time.time(),'directory':str(directory),'read_only':True,'new_physics_steps':0}
receipt.write_text(json.dumps(v,indent=2),encoding='utf-8');print(json.dumps(v))
