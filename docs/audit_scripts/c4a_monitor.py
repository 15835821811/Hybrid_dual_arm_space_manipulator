"""Compact read-only progress; no inference, physics or file writes."""
from datetime import datetime,timezone
import json
from pathlib import Path
import sys

run=Path(sys.argv[1]);streams=[];events=[]
for path in sorted((run/'command_logs').glob('*')):
    out=path/'stdout.txt';lines=out.read_text(encoding='utf8').splitlines() if out.exists() else []
    own=[]
    for line in lines:
        if line.startswith('{') and ('C1_CANDIDATE_TERMINAL' in line or 'C3_FINAL_ACTUAL' in line):
            try:own.append(json.loads(line))
            except json.JSONDecodeError:pass
    receipt=path/'receipt.json'
    streams.append(dict(command=path.name,completed=receipt.exists(),events=len(own),
        exit_code=json.loads(receipt.read_text())['exit_code'] if receipt.exists() else None,
        last=own[-1] if own else None))
    events.extend(own)
print(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(),
    execution_complete=(run/'execution_complete.json').exists(),
    total_candidate_events=sum(e['event']=='C1_CANDIDATE_TERMINAL' for e in events),
    total_actual_events=sum(e['event']=='C3_FINAL_ACTUAL' for e in events),streams=streams),ensure_ascii=False))
