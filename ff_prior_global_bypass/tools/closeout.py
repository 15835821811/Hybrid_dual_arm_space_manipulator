"""Read saved evidence and perform the single authorized no-work resume check.

Never constructs a robot, propagates physics or changes the frozen source.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from ff_prior.storage import read_json
from ff_prior_pilot_train.storage import atomic_json
from ff_prior.coupled_predictor import pose_errors

RUN=ROOT/'artifacts/v7_1/global_bypass_001'
EVIDENCE=ROOT/'ff_prior_global_bypass/evidence'


def distribution(values):
    a=np.asarray(values,float)
    if not len(a):return {'count':0,'mean':None,'median':None,'p95':None,'worst':None,'minimum':None}
    return {'count':len(a),'mean':float(a.mean()),'median':float(np.median(a)),
        'p95':float(np.percentile(a,95)),'worst':float(a.max()),'minimum':float(a.min())}


def endpoint(ee,goals):
    pos,rot=pose_errors(np.asarray(ee),goals)
    return {'Q':float(max(np.max(pos/.005),np.max(rot/np.deg2rad(1)))),
        'position_mm':float(max(pos)*1000),'rotation_deg':float(np.rad2deg(max(rot)))}


def statistics():
    ident=read_json(RUN/'run.json')['identity'];goals=ident['selection']['entries'][0]['task']['condition']['goals']
    rows=[];goals_rows=[];units=[];edge_counts=Counter();grid_counts=Counter();pairs=Counter()
    for spec in ident['selection']['units']:
        d=RUN/'units'/spec['id'];result=read_json(d/'result.json')
        for result_row in result['candidates']:
            pd=d/'candidates'/f'{result_row["id"]:02d}';td=pd/'tree'
            tr=read_json(td/'result.json');attempts=read_json(td/'attempts.json')
            with np.load(td/'nodes.npz') as a:
                node_scores=[endpoint(ee,goals) for ee in a['ee']]
                best=min(node_scores,key=lambda x:x['Q'])
                chosen=node_scores[tr['best_node']]
                node_time_max=float(max(a['elapsed']))
                base_not_evaluated_here=True
            with np.load(pd/'goal_seed/result.npz') as a:seed_score=endpoint(a['ee'],goals)
            erows=[read_json(p) for p in sorted((td/'edges').glob('*/report.json'))]
            ec=Counter()
            for er in erows:
                ec['propagations']+=1;grid_counts[er['intervals']]+=1
                ec['accepted']+=int(er['accepted']);ec['complete']+=int(er['complete'])
                ec['numerical_failures']+=int(not er['numerical_ok'])
                if er['minimum_clearance_m']<0:
                    ec['penetrating']+=1;p=er['minimum_pair'];pairs[(p['geom_a'],p['geom_b'])]+=1
                elif not er['accepted']:ec['refinement_only']+=1
            edge_counts.update(ec)
            connections=[]
            for gp in sorted((td/'goal_connections').glob('*/iterations.json')):
                records=read_json(gp);accepted=[t for r in records for t in r['trials'] if t.get('accepted')]
                best_pose=min((endpoint(t['ee'],goals) for t in accepted),key=lambda x:x['Q'],default=None)
                row={'unit':spec['id'],'candidate':result_row['id'],'method':spec['method'],
                    'node':int(gp.parent.name),'records':len(records),'accepted_updates':len(accepted),
                    'trials':sum(len(r['trials']) for r in records),
                    'edge_pass_trials':sum(t['edge_pass'] for r in records for t in r['trials']),
                    'last_no_accepted_step':not records[-1]['accepted'],
                    'initial_Q':records[0]['Q'],'last_before_Q':records[-1]['Q'],
                    'best_accepted_pose':best_pose,'path':gp.relative_to(ROOT).as_posix()}
                connections.append(row);goals_rows.append(row)
            rows.append({'unit':spec['id'],'method':spec['method'],'seed':spec['seed'],
                'candidate':result_row['id'],'proposal_seed':spec['proposal_seeds'][result_row['id']],
                'status':result_row['status'],'stop':tr['stop'],'attempts':tr['attempts'],
                'nodes_excluding_root':tr['nodes']-1,'maximum_node_time_s':node_time_max,
                'time_reserve_rejections':sum(a.get('reason')=='TIME_RESERVE' for a in attempts),
                'edge_counts':dict(ec),'seed_pose':seed_score,'best_tree_world_pose':best,
                'q_nearest_selected_pose':chosen,'goal_connection_records':len(connections),
                'sampling_counts':dict(Counter(a['sampling'] for a in attempts)),
                'fit_present':(pd/'fit.npz').exists(),'eligible':result_row['eligible']})
        ur=[r for r in rows if r['unit']==spec['id']]
        units.append({'unit':spec['id'],'method':spec['method'],'seed':spec['seed'],
            'attempts':sum(r['attempts'] for r in ur),'nodes':sum(r['nodes_excluding_root'] for r in ur),
            'coupled':read_json(d/'calls.json')['coupled'],
            'best_world_Q':min(r['best_tree_world_pose']['Q'] for r in ur),
            'recorded_goal_sessions':sum(r['goal_connection_records'] for r in ur),'qualified':result['qualified']})
    methods={}
    for method in ident['protocol']['methods']:
        rr=[r for r in rows if r['method']==method];gg=[r for r in goals_rows if r['method']==method]
        methods[method]={'candidate_denominator':15,'qualified':sum(r['eligible'] for r in rr),
            'attempts':sum(r['attempts'] for r in rr),'nodes':sum(r['nodes_excluding_root'] for r in rr),
            'time_reserve_rejections':sum(r['time_reserve_rejections'] for r in rr),
            'coupled':sum(u['coupled'] for u in units if u['method']==method),
            'goal_sessions_recorded':len(gg),'goal_iterations':sum(g['records'] for g in gg),
            'goal_accepted_updates':sum(g['accepted_updates'] for g in gg),
            'goal_sessions_no_final_descent':sum(g['last_no_accepted_step'] for g in gg)}
        for stage in ('seed_pose','best_tree_world_pose','q_nearest_selected_pose'):
            methods[method][stage]={metric:distribution([r[stage][metric] for r in rr])
                for metric in ('Q','position_mm','rotation_deg')}
        methods[method]['accepted_goal_session_best']={metric:distribution(
            [g['best_accepted_pose'][metric] for g in gg if g['best_accepted_pose'] is not None])
                for metric in ('Q','position_mm','rotation_deg')}
    m0=np.load(RUN/'m0/propagation_check.npz')
    files=[p for p in RUN.rglob('*') if p.is_file()]
    result={'run':RUN.relative_to(ROOT).as_posix(),'status':read_json(RUN/'status.json'),
        'budget':read_json(RUN/'budget.json'),'methods':methods,'units':units,
        'edge_counts':dict(edge_counts),'edge_grids':dict(grid_counts),
        'collision_pairs':[{'geoms':list(k),'count':v} for k,v in pairs.most_common()],
        'tree_stops':dict(Counter(r['stop'] for r in rows)),
        'recorded_goal_sessions':len(goals_rows),'goal_iterations':sum(g['records'] for g in goals_rows),
        'goal_trials':sum(g['trials'] for g in goals_rows),'goal_edge_pass_trials':sum(g['edge_pass_trials'] for g in goals_rows),
        'goal_accepted_updates':sum(g['accepted_updates'] for g in goals_rows),
        'goal_sessions_no_final_descent':sum(g['last_no_accepted_step'] for g in goals_rows),
        'm0_max_qpos_difference':float(np.max(abs(m0['joined_qpos']-m0['whole_qpos']))),
        'm0_max_qvel_difference':float(np.max(abs(m0['joined_velocity']-m0['whole_velocity']))),
        'core_files':len(files),'core_bytes':sum(p.stat().st_size for p in files),
        'new_physics_in_analysis':0,'scope':'saved evidence only; best prefix is not a complete solution',
        'goal_session_denominator_note':'only sessions with saved iteration records; do not assert all function invocations recorded'}
    atomic_json(EVIDENCE/'statistics.json',result);atomic_json(EVIDENCE/'candidate_statistics.json',rows)
    atomic_json(EVIDENCE/'goal_connection_statistics.json',goals_rows)
    with (EVIDENCE/'unit_metrics.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(units[0]));writer.writeheader();writer.writerows(units)
    print(json.dumps({k:result[k] for k in ('methods','edge_counts','tree_stops','recorded_goal_sessions',
        'goal_iterations','goal_accepted_updates','goal_sessions_no_final_descent','core_files','core_bytes')},ensure_ascii=False,indent=2))


def inventory():
    rows={}
    for p in sorted(RUN.rglob('*')):
        if not p.is_file():continue
        h=hashlib.sha256()
        with p.open('rb') as f:
            for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
        rows[p.relative_to(RUN).as_posix()]={'size':p.stat().st_size,'sha256':h.hexdigest()}
    return rows


def resume_once():
    marker=EVIDENCE/'completed_resume_attempt.json';receipt=EVIDENCE/'completed_resume.json'
    if marker.exists() or receipt.exists():raise RuntimeError('ONCE_ONLY_RECEIPT_ALREADY_EXISTS')
    assert read_json(RUN/'status.json')['status'] in ('FINISHED','FINISHED_WITH_BLOCKED_ITEMS')
    before=inventory();atomic_json(EVIDENCE/'core_manifest.json',before)
    atomic_json(marker,{'started_at':time.time(),'command':[sys.executable,'-m','ff_prior_global_bypass.job','resume'],
        'no_repeat':True,'core_file_count':len(before)})
    result=subprocess.run([sys.executable,'-m','ff_prior_global_bypass.job','resume'],cwd=ROOT,
        capture_output=True,text=True,encoding='utf-8')
    after=inventory();response=json.loads(result.stdout) if result.returncode==0 else None
    value={'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr,
        'response':response,'before_files':len(before),'after_files':len(after),
        'core_unchanged':before==after,'new_physics':0 if before==after and response and
            response.get('resume')=='NO_WORK_ALREADY_COMPLETE' else None,
        'checked_at':time.time(),'pass':bool(result.returncode==0 and before==after and
            response.get('resume')=='NO_WORK_ALREADY_COMPLETE')}
    atomic_json(receipt,value);print(json.dumps(value,ensure_ascii=False,indent=2))
    if not value['pass']:raise SystemExit(1)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['statistics','resume-once'])
    a=parser.parse_args();statistics() if a.action=='statistics' else resume_once()
