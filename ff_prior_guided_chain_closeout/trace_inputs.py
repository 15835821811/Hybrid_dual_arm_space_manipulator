from pathlib import Path
import hashlib,json,datetime
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'artifacts/v7_1/guided_diffusion_chain_001'
TRACE=ROOT/'paper/review-traces/experiment-result-to-claim/20261026_guided_chain'
files=[p for p in RUN.rglob('*') if p.is_file()]
files += [p for p in (ROOT/'ff_prior_guided_chain').rglob('*') if p.is_file() and '__pycache__' not in p.parts]
files += [p for p in (ROOT/'ff_prior_guided_chain_closeout').rglob('*') if p.is_file() and '__pycache__' not in p.parts]
files += [TRACE/'prompt.md']
records=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size,modified_utc=datetime.datetime.fromtimestamp(p.stat().st_mtime,datetime.timezone.utc).isoformat()) for p in sorted(set(files))]
(TRACE/'inputs.json').write_text(json.dumps(dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=records,scope='New run only. Bound source/input identities listed additionally in frozen.json; no old raw audit.',paper_claim_audit='unavailable',writing_verdict='provisional'),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(review_input_files=len(records))))
