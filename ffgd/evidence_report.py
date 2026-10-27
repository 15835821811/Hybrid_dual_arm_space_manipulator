"""Inventory every FFGD run and candidate, without merging version successes."""
import collections
import hashlib
import json
from pathlib import Path
from .orchestration import ROOT,atomic_json,read_state,utc


def inventory(runs,exclude=None):
    runs=Path(runs);units=[];planning=[];issues=[]
    for folder in sorted(runs.iterdir()):
        if not folder.is_dir() or folder==exclude:continue
        def read(name,default=None):
            path=folder/name
            if not path.exists():return default
            try:return json.loads(path.read_text(encoding='utf-8-sig'))
            except (ValueError,OSError) as error:
                issues.append(dict(file=str(path),error=str(error)));return default
        manifest=read('manifest.json')
        if manifest is None:continue
        command=manifest.get('command',[])
        units.append(dict(run=folder.name,status=manifest.get('status','UNKNOWN'),command=command,
                          wall_seconds=manifest.get('wall_seconds'),source_hash=manifest.get('source_hash')))
        if len(command)<4 or command[3] not in ('plan','benchmark'):continue
        metadata=read('planning_manifest.json',{});result=read('planning_result.json',{})
        supplemental=read('failure_denominator.json',{})
        seeds=metadata.get('budget',{}).get('seeds',[])
        requested=result.get('requested_candidates',supplemental.get('requested_candidates',manifest.get('budget',{}).get('candidates')))
        if requested is None:
            issues.append(dict(run=folder.name,error='missing requested candidate denominator'));continue
        records={row['index']:row for row in result.get('records',[])}
        candidates=[]
        for index in range(requested):
            row=records.get(index) or read(f'candidate_{index:03d}/candidate.json',{})
            candidates.append(dict(index=index,seed=row.get('seed',seeds[index] if index<len(seeds) else supplemental.get('seed')),
                status=row.get('status','NOT_RUN'),sampled=row.get('sampled',False),validated=row.get('validated',False),
                model_valid=row.get('status')=='PLAN_VALID',network_calls=row.get('observed_network_calls'),
                failure_gates=row.get('validation',{}).get('failures'),error=row.get('error'),
                candidate_wall_seconds=row.get('candidate_wall_seconds'),cumulative_wall_seconds=row.get('cumulative_wall_seconds',row.get('elapsed_seconds'))))
        if len(records)>requested:issues.append(dict(run=folder.name,error='candidate records exceed requested denominator'))
        version=dict(source_hash=manifest.get('source_hash'),method=metadata.get('method',manifest.get('budget',{}).get('method')),checkpoint=metadata.get('checkpoint',{}).get('sha256'),
                     budget=metadata.get('budget'),guide_weights=metadata.get('guide_weights'),physics_backend=metadata.get('physics_backend'),
                     collision_aggregation=metadata.get('collision_aggregation','pair_mean'),joint_peak_penalty=metadata.get('joint_peak_penalty',False))
        group=hashlib.sha256(json.dumps(version,sort_keys=True).encode()).hexdigest()
        planning.append(dict(run=folder.name,unit_status=manifest['status'],result_status=result.get('status','NOT_RUN' if manifest['status']=='RUNNING' else 'INCOMPLETE_OR_SETUP_FAILURE'),
            group=group,version=version,scope='FIXED_SEED_DIAGNOSTIC' if requested<5 else 'MULTI_NOISE_TASK_EVALUATION',
            requested=requested,sampled=sum(c['sampled'] for c in candidates),model_valid=sum(c['model_valid'] for c in candidates),
            selected_index=result.get('selected_index'),candidates=candidates,supplemental_failure=supplemental or None))
    groups=[]
    for key in sorted({r['group'] for r in planning}):
        members=[r for r in planning if r['group']==key]
        groups.append(dict(group=key,runs=[r['run'] for r in members],scope=members[0]['scope'],
                           requested=sum(r['requested'] for r in members),sampled=sum(r['sampled'] for r in members),
                           model_valid=sum(r['model_valid'] for r in members)))
    return dict(consistent=not issues,issues=issues,units=units,unit_status_counts=dict(collections.Counter(r['status'] for r in units)),
                planning_runs=planning,version_groups=groups,
                limitation='Inventory only. Diagnostic versions are separate; no pooled success rate or milestone promotion. Missing per-candidate timing in old runs stays null.')


def run(output):
    report=inventory(ROOT/'artifacts/ffgd/runs',exclude=Path(output));state=read_state()
    report.update(as_of=utc(),overall_status=state['overall_status'],current_milestone=state['current_milestone'],
                  architecture_status=state['architecture_status'],engineering_status=state['engineering_status'],
                  learning_benefit_status=state['learning_benefit_status'],milestones=state['milestones'])
    atomic_json(output/'inventory.json',report)
    return dict(passed=report['consistent'],scope='evidence inventory consistency only; no engineering completion claim',
                unit_count=len(report['units']),planning_run_count=len(report['planning_runs']),issues=report['issues'])
