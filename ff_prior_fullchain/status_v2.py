"""Read-only routing overlay; preserves every archived experiment dependency."""
import argparse
import json
from . import job


def snapshot():
    result=job.snapshot();control=job.read(job.PACKAGE/'control.json')
    if result.get('worker',{}).get('status') in job.TERMINAL:
        if control.get('goal_status') in ('AWAITING_SCOPE_AUTHORIZATION','BLOCKED_PENDING_SCOPE_AUTHORIZATION'):
            result.update(action='AWAIT_SCOPE_AUTHORIZATION_NO_AUTOMATIC_EXPERIMENT',
                decision_document=control.get('decision_document'),
                engineering_complete=False,active_physics_worker=False)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['status','next','report'],default='status',nargs='?')
    a=p.parse_args()
    if a.action=='report':
        c=job.read(job.PACKAGE/'control.json');report=job.ROOT/c['active_report']
        print(report.read_text(encoding='utf-8') if report.exists() else '当前报告未完成；不得将状态或计划写成通过。')
        if c.get('decision_document'):print('\n下一步决策：'+str(job.ROOT/c['decision_document']))
    else:print(json.dumps(snapshot(),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
