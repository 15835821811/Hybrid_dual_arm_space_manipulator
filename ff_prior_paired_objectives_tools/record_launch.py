"""Record an observed launch without treating pending experiments as passed."""
from pathlib import Path
import shutil,datetime
from ff_prior.storage import read_json,file_digest
from ff_prior_coverage_train.storage import atomic_json
from ff_prior_coverage_train.job import status
from ff_prior_paired_objectives.protocol import ROOT,DEFAULT_OUT

if __name__=='__main__':
    state=status(DEFAULT_OUT)
    assert state['status']=='RUNNING' and state['owner_alive']
    stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    proof=read_json(ROOT/'ff_prior_paired_objectives/evidence/preflight.json')
    atomic_json(ROOT/'ff_prior_paired_objectives/evidence/launch.json',{
        'source_commit':'ded2d537','observed_status':state,'launcher':read_json(DEFAULT_OUT/'launcher.json'),
        'preflight_sha256':file_digest(ROOT/'ff_prior_paired_objectives/evidence/preflight.json'),
        'tests':13,'formal_results':'PENDING','heartbeat_id':'automation','heartbeat_status':'ACTIVE'})
    base=(ROOT/'ff_prior_paired_objectives/docs/TRACKER_20261019.md').read_text(encoding='utf-8')
    text=base.replace('IMPLEMENTED_TESTING','PASS / WORKER_RUNNING').replace('尚不声称正式通过','13/13测试；178645来源/231源码核验PASS；正式科学结果仍待测')
    text+='\n启动记录：源码ded2d537；worker PID '+str(state['pid'])+'，观察阶段 '+state['stage']+'，心跳存活。独立输出artifacts/v7_1/paired_objectives_smoke_001；接续automation已恢复，每30分钟只检查状态，运行正常时安静等待，结束后审查与报告。\n'
    version=ROOT/'paper'/f'EXPERIMENT_TRACKER_{stamp}.md';version.write_text(text,encoding='utf-8')
    shutil.copyfile(version,ROOT/'paper/EXPERIMENT_TRACKER.md')
    (ROOT/'ff_prior_paired_objectives/docs/LAUNCH.md').write_text(
        '# 本轮启动记录\n\n2026-10-19：源码ded2d537，13/13实际软件测试通过；178645来源文件/231 Python源码预检PASS，原冻结任务hash保持。\n\n'
        f'后台worker {state["pid"]}已启动且心跳正常，观察阶段{state["stage"]}。正式单步诊断/四组训练/DEV成绩均待实测。\n\n'
        '当前入口：`.\\ff_prior_paired_objectives\\job.ps1 status|report|resume`。完成后由已有接续做独立审查、完成恢复核查和中文ANALYSIS。\n\n'
        '统一研究目标已重整为DIFFUSION_GOAL_CURRENT.md，历史全文保留。原1024槽、pilot、TEST保持未启动。应用旧目标卡片仍暂停，工具不能原地恢复或改写；本地后台及完成接续已恢复执行。\n',encoding='utf-8')
    with (ROOT/'MANIFEST.md').open('a',encoding='utf-8') as f:
        f.write(f'\n- {stamp}: {version.relative_to(ROOT).as_posix()}; ff_prior_paired_objectives/docs/LAUNCH.md; evidence/launch.json; RUNNING, scientific results pending.\n')
    print({'status':'RECORDED','worker_pid':state['pid'],'stage':state['stage'],'tracker':version.name})
