"""Write final Chinese analysis from completed, reviewed evidence only."""
from pathlib import Path
import datetime,shutil
from ff_prior.storage import read_json,file_digest
from ff_prior_coverage_train.storage import atomic_json
from ff_prior_paired_objectives.protocol import ROOT,DEFAULT_OUT,ARMS
from .summarize_results import summarize

if __name__=='__main__':
    out=DEFAULT_OUT;doc=ROOT/'ff_prior_paired_objectives/docs';ev=ROOT/'ff_prior_paired_objectives/evidence/completed_001'
    review=ROOT/'paper/review-traces/experiment-result-to-claim/20261019_paired_run01'
    r=summarize();state=r['status'];verdict=read_json(review/'verdict.json');resume=read_json(ev/'resume_noop.json')
    assert state['status']=='FINISHED_WITH_BLOCKED_ITEMS' and r['independent_verification']['status']=='PASS'
    assert r['TRAIN_selection']['status']=='BLOCKED' and resume['status']=='PASS'
    assert not (out/'candidates').exists() and not (out/'executions').exists()
    ev.mkdir(parents=True,exist_ok=True)
    atomic_json(ev/'posthoc_diagnosis.json',r)
    compact=['status.json','summary.json','TRAIN_selection.json','training_verification.json','independent_verification.json','runtime_cache_audit.json']
    for name in compact:shutil.copyfile(out/name,ev/name)
    for name in ['warm']+list(ARMS):
        dst=ev/'probes'/f'{name}.json';dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(out/'probes'/f'{name}.json',dst)
    atomic_json(ev/'release_decision.json',{'status':'COMPLETE_WITH_DEV_BLOCKED','new_development_default':False,
        'keep_development_default':'qualified_origin_learn_002/learning/goal_loss',
        'D_both':'research candidate only; modest TRAIN improvement, no DEV qualification',
        'new_training_rows':0,'new_native_executions':0,'TEST':False,'pilot':False,'broad1024':False,
        'full_goal_achieved':False,'review_verdict':verdict,'raw_evidence_root':str(out),
        'raw_evidence_hashes':{n:file_digest(out/n) for n in compact}})
    p=r['probes'];s=r['TRAIN_selection'];minutes=state['elapsed_s']/60
    lines=['# 四组配对训练：真实终点误差有所改善，但未达到DEV准入门','',
        f'2026-10-19。本轮后台已结束，耗时 {state["elapsed_s"]:.3f} 秒（{minutes:.2f} 分钟），状态 FINISHED_WITH_BLOCKED_ITEMS。进程已退出，原始审计PASS；BLOCKED指TRAIN改善门未过，故DEV未启动，并非程序崩溃。',
        '源码ded2d537，启动记录47e30678；独立输出artifacts/v7_1/paired_objectives_smoke_001。13/13软件测试与178645来源/231源码预检通过。','',
        '## 四组完成了什么','',
        '四组全部完成6000次更新，各384000次样本抽取，13838条TRAIN全部实际参与；VAL48不用于梯度或模型选择。数据、起始13774模型、codec、批次和噪声序列相同，独立重建6000步一致；固定最终权重重载通过。一个训练种子的配对实验，不能据此宣称跨种子稳定优势。',
        'A仅v去噪；B加0.05局部位姿G；C每50步加0.1真实世界终点项；D保留局部G并叠加同样世界项。C/D各120次全102维中央差分，合计50880次传播；全部240个梯度前置网络权重已保存，原始审计逐个重建当前预测。世界项以完整自由漂浮传播计算，不是教师附近线性近似。',
        '本轮没有生成新数据，也没有新增TRAIN力矩执行。13838条TRAIN中包含的64条合格修正和16次预选执行属于上一轮，不能重复记成本轮成果。','',
        '## 固定32例TRAIN探针结果','',
        '|方法|相对旧模型的配对损失中位比↓|双臂较大位置误差中位数mm↓|双臂较大姿态误差中位数°↓|位姿门通过/32|损失改善/32|',
        '|---|---:|---:|---:|---:|---:|']
    for name,label in [('warm','旧13774起始模型'),('A_v','A 仅去噪'),('B_local','B 去噪＋局部位姿'),('C_world','C 去噪＋真实终点'),('D_both','D 去噪＋局部位姿＋真实终点')]:
        q=p[name]['all'];lines.append(f'|{label}|{q["loss_ratio_median"]:.6f}|{q["position_max_arm_median_mm"]:.3f}|{q["rotation_max_arm_median_degrees"]:.3f}|{q["pose_pass"]}|{q["loss_improved"] if name!="warm" else "—"}|')
    lines+=['',
        '损失为双臂位置/5mm及旋转/1°的平方和除4；中位比先逐例after/before再取中位，不是两组中位损失相除。位置和姿态分别取每例双臂中的较大值，再汇总。位姿门只表示双臂均≤5mm/1°，不等于全机MODEL_VALID或执行通过。',
        '这些是对已有TRAIN教师轨迹加噪后的一次网络去噪预测，再以512间隔完整自由漂浮传播计算的世界终点；不是16步DDIM从噪声生成的完整采样，也不是DEV或真实力矩执行结果。32探针来自16个TRAIN目标各两个噪声档，不能当作32个独立任务；曾在上一轮使用且本轮用于选组，不是未见泛化证据。',
        'D组相对旧模型综合损失中位比0.914089，约改善8.59%；相对B为0.853648，约改善14.64%。但D姿态误差中位数从1.117°升至1.325°，位姿通过从5/32降为4/32。平均意义的局部改善尚未转化为稳定达门能力。','',
        '进一步逐例核对：D位置误差17/32改善、姿态14/32改善；其中5例综合损失下降但姿态误差上升。旧模型5个过门探针中保留4个、丢失1个，没有新增过门探针。训练的平方和允许分项取舍，最终验收要求所有位姿条件同时满足；这解释了为何综合损失下降不能直接当成通过率提升，但尚不能据此判定全部退化的单一原因。','',
        '## DEV门为什么没有通过','',
        '预注册要求C或D同时满足：相对旧模型的配对损失中位比≤0.8，且相对B≤1；四组实际训练、单步诊断和原始训练审计通过。C为4.710018/3.576178，两项均失败；D为0.914089/0.853648，仅“不劣于B”通过。selected_arm=null，DEV依赖BLOCKED。',
        '因此本轮原DEV4＋已用8开发任务的720候选和最多144次执行重放均为未运行，而非0成功率；没有新的DEV收益、物理执行、DDIM稳定性或样条对照结论。未修改准入门，也未利用DEV补选模型。','',
        '## 差异和原因：可确认与待验证','',
        '1. 真实梯度的数值链路在测试案例内有效。8/8单步参数方向检查通过，最大相对偏差0.000066632（约0.0067%）；三个单位负梯度小步均在全部8例降低真实终点损失。不能用本轮退化断言微重力建模错误或梯度符号错误。',
        '2. 局部位姿监督在本配对实验中明显抑制退化。A的损失比3.349而B为1.048；C为4.710而D为0.914。相同数据、初始化和随机序列下，保留局部项是关键差别；但B自身也未明显胜过旧模型。此结论限定于本训练种子和这些TRAIN探针。',
        '3. 真实终点项的效果依赖组合。C比A更差；D相对B有小幅正向变化，不能概括为“真实终点损失没用”。上一轮删除每批局部G，同时只每50步注入一个世界项，不是等价替换。',
        '4. 优化尺度仍需诊断。C/D的120次世界事件全部触发clip1。C事件参数梯度范数中位64.482/最大1059.362；D中位30.321/最大449.087。加权世界项中位范数C64.467、D30.230；对应v约0.601、0.612，D局部项约0.488。世界事件中该单样本项明显主导更新。裁剪本身不是错误，也不能单凭这些数值证明退化由裁剪造成；权重、注入频率、AdamW状态及噪声覆盖仍需单因素区分。',
        '5. 高噪声的收益更弱。D在t500的配对损失中位比0.884322，t999为0.987454，最高噪声仅约1.25%改善。当前120事件沿用变体与两个噪声档的绑定，不能证明其他噪声或完整采样链已稳定。单步local/world梯度夹角接近同向，而训练事件两项来自不同样本；不能将训练中的低余弦直接解释为同一目标上的梯度冲突。',
        '单步fresh AdamW：A/B/C/D分别1/8、8/8、6/8、5/8改善。8/8数值下降与组合优化器并非8/8下降可以同时成立；归一化负梯度步与AdamW预条件更新是不同操作。','',
        '## 下一步及模型处置','',
        '保留旧13774模型作为已验证开发基准。D作为下一轮研究候选保留，四组所有权重和失败结果不覆盖；不因为D训练探针略好就替换正式基准。',
        '下一轮优先保留每批局部G，只在TRAIN上诊断世界项优化尺度：先独立改变世界项权重或参数步长，并测更新前后的真实终点变化、局部/世界损失及回归；随后再单独研究注入频率和噪声/变体覆盖。需另冻有界协议，避免同时改变多因素。本轮没有启动这些新实验。继续使用原门，改善足够后才做原冻结DEV同预算复核。',
        '原1024槽扩库、pilot、study、strict_multimodal和TEST均未启动，TEST未读取。数据总量不是本轮直接瓶颈的已证因果；现有合格数据仍可用，当前缺口是学习与优化尚不能稳定达到目标。','',
        '## 审查、恢复和四项结论','',
        f'新的独立评审见paper/review-traces/experiment-result-to-claim/20261019_paired_run01/；整体判定：{verdict.get("claim_supported","见verdict.json")}，置信度：{verdict.get("confidence","见verdict.json")}。C1实现和所测梯度链路支持，C2的预注册改善/DEV资格与优越性不支持。没有PAPER_CLAIM_AUDIT时，下游论文表述仍为provisional，不影响已核实原始数值。',
        f'完成resume实测NO_WORK_ALREADY_COMPLETE；{len(resume["before"])}项核心文件及模型hash均不变，没有新增训练或物理调用。',
        '|结论层次|本轮结论|','|---|---|',
        '|软件与真实训练实现|通过；13测试、四组6000更新、240当前预测梯度和原始审计|',
        '|模型数据资格|继承13838TRAIN/48VAL；本轮新增0，既有验证等级不变|',
        '|真实力矩执行/独立重放|本轮正式实验未运行；软件回归测试不计为DEV成绩|',
        '|Diffusion收益|D的TRAIN综合误差小幅改善；目标通过率未改善，DEV门未过，优越性未成立|',
        '', '长期目标未完成。本轮收尾完成后暂停完成接续，避免重复工作。',
        '', '```powershell',r'.\ff_prior_paired_objectives\job.ps1 status',r'.\ff_prior_paired_objectives\job.ps1 report',r'.\ff_prior_paired_objectives\job.ps1 resume','```']
    (doc/'ANALYSIS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    tracker='''# 四组配对训练已完成，DEV改善门未通过

|阶段|状态|证据与边界|
|---|---|---|
|软件/来源|PASS|13/13测试，178645来源/231源码|
|TRAIN单步诊断|PASS|8/8参数梯度方向；三个小步均下降|
|四组真实训练|PASS|各6000更新、384000抽取，13838TRAIN全部使用，同权重和序列|
|真实终点梯度审计|PASS|C/D各120，共240个当前预测由保存权重重建|
|固定TRAIN探针|部分改善|A3.348868、B1.048363、C4.710018、D0.914089；D位姿4/32，旧5/32|
|原冻结DEV复核|BLOCKED_NOT_RUN|C/D均未满足相对warm≤0.8；没有候选和执行|
|来源/原始结果/独立审查|完成|见本轮ANALYSIS和新的review-traces，收益边界保留|
|完成resume|PASS|NO_WORK_ALREADY_COMPLETE，9项核心hash不变|
|旧1024/pilot/TEST|NOT_STARTED|用户当前范围禁止|
|长期目标|未完成|下一优先：保留局部G，TRAIN有界优化尺度诊断；尚未启动新实验|

报告：ff_prior_paired_objectives/docs/ANALYSIS.md。后台2497.297秒，旧模型/数据/结果/用户修改保留；本轮没有新增数据和正式力矩执行。
'''
    version=ROOT/'paper'/f'EXPERIMENT_TRACKER_{stamp}.md';version.write_text(tracker,encoding='utf-8');shutil.copyfile(version,ROOT/'paper/EXPERIMENT_TRACKER.md')
    goal=ROOT/'DIFFUSION_GOAL_CURRENT.md';old=goal.read_text(encoding='utf-8')
    archive=doc/'history'/f'GOAL_before_closeout_{stamp}.md';archive.write_text(old,encoding='utf-8')
    update='''## 2026-10-19本轮已完成：四组配对学习，DEV准入BLOCKED

paired_objectives_smoke_001已退出，2497.297秒；原始审计PASS。8/8参数单步检查通过，四组各6000更新/384000抽取/13838TRAIN全参与，C/D各120个真实102维当前预测梯度，240个保存权重重建通过。A/B/C/D固定TRAIN探针损失比分别3.348868/1.048363/4.710018/0.914089。D相对B为0.853648，有小幅改善；但相对旧模型未达≤0.8，位姿通过4/32低于旧5/32。因此DEV未运行，最终TEST和原1024扩库仍未启动。

下一优先保留每批局部G，先另冻TRAIN优化尺度诊断，区分世界项权重/步长与频率/噪声覆盖；不得跳过DEV门或因此直接扩库。本轮新增数据0、正式物理执行0；旧模型继续开发基准，D保留研究候选。完成resume无重复，独立结论见ff_prior_paired_objectives/docs/ANALYSIS.md。长期目标尚未完成，本轮完成接续收尾后暂停。

'''
    text=old.replace('## 最终目标',update+'## 最终目标',1)
    text=text.replace('本轮最高优先，独立后台paired_objectives_smoke_001','本轮已完成并审计；下一有界TRAIN优化尺度协议待冻结')
    text=text.replace('通过前置门才运行；未过明确BLOCKED，不强行消耗DEV','本轮BLOCKED：D为0.914089>0.8；未启动DEV')
    goal.write_text(text,encoding='utf-8');shutil.copyfile(goal,ROOT/'ff_prior_feasible/docs/CORE_GOAL.md')
    with (ROOT/'findings.md').open('a',encoding='utf-8') as f:f.write('\n## 2026-10-19 paired_objectives_smoke_001\n\n四组同数据/初始化/序列训练完成。D TRAIN损失中位比0.914089，门要求0.8；位姿4/32对旧5/32，DEV未运行。局部项抑制退化；世界项与局部组合有小幅收益，优化尺度/频率原因待验证。独立评审：'+str(verdict)+'。下一路由：保留局部G，TRAIN优化尺度诊断，不启动扩库或TEST。\n')
    with (ROOT/'MANIFEST.md').open('a',encoding='utf-8') as f:f.write(f'\n- {stamp}: ff_prior_paired_objectives/docs/ANALYSIS.md; evidence/completed_001/; {version.relative_to(ROOT).as_posix()}; new independent review 20261019_paired_run01. COMPLETE_WITH_DEV_BLOCKED.\n')
    print({'status':'WRITTEN','tracker':version.name,'report':str(doc/'ANALYSIS.md')})
