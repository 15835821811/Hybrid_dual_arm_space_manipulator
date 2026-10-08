# Independent result-to-claim review — V6.4-C.2

**claim_supported: partial; confidence: medium.** Finite engineering delivery and real learned initialization are complete. Local eight-slot A coverage improves over R8; overall superiority beyond retrieval and quality-preserving amortization are not established. Recommended `learning_benefit_established_in_pilot=NOT_ESTABLISHED`, default `retain_C1_rule`.

This review used only terminal evidence in `E:\v64c2\v6_4\output\preference_warmstart_20261008_01`; no training, inference, search, physics, or raw evidence was run or changed. No paper claim audit exists: downstream paper claims remain provisional, which does not block finite delivery.

## Engineering and data

Validation reports48 new teacher slots,112 TEST slots,32 logical actual slots,12 unique actual runs, zero tool errors,1819440 prediction main steps and162000 actual main steps. All32 slot hashes and all sealed-selection manifest hashes were independently checked and match. Ten complete aliases match their source Task/plan/alias identity; ten NO_PLAN slots retain zero steps. Selections were globally sealed at02:09:23.075016 UTC before actual began at02:10:08.986294 UTC. Evidence: [validation/validation.json](E:/v64c2/v6_4/output/preference_warmstart_20261008_01/validation/validation.json), [actual_complete.json](E:/v64c2/v6_4/output/preference_warmstart_20261008_01/actual_complete.json), [sealed_selections/all_selections.json](E:/v64c2/v6_4/output/preference_warmstart_20261008_01/sealed_selections/all_selections.json).

Dataset counts only:96 unique physical candidates (48 historical,48 new);10 VALIDATED_EXECUTION,64 PREDICTED_COMPLETE,22 FAILED_OR_INCOMPLETE. There are77 preference/family label views from65 physical candidates:52 TRAIN views and25 VAL views, including4 zero views. TRAIN contains3 mothers/6 Tasks, VAL1 mother/2 Tasks, TEST2 mothers/4 Tasks. Formal A-v1 and B-v2 injection conditions have TRAIN/VAL labels19/8 and10/12 respectively, both supported across3 TRAIN mothers. Training performed4000 updates/128000 exposures; the selected checkpoint experienced250 updates, with16 VAL checkpoints/128 DDIM samples. TEST generated8 raw seeds,7 legal; no repairs or resampling appear in proposal diagnostics. Evidence: [dataset/dataset_manifest.json](E:/v64c2/v6_4/output/preference_warmstart_20261008_01/dataset/dataset_manifest.json), [model/training_summary.json](E:/v64c2/v6_4/output/preference_warmstart_20261008_01/model/training_summary.json), [summary.json](E:/v64c2/v6_4/output/preference_warmstart_20261008_01/summary.json) and `benchmark_search/<Task>/D/planning/<Task>/proposals.json`.

## Actual coverage and predeclared quality

|Endpoint|A full Task + five gates|B full Task + five gates +30mm|A NO_PLAN|B NO_PLAN|
|---|---:|---:|---:|---:|
|R8|2/4|2/4|2|2|
|R12|4/4|2/4|0|2|
|N8|4/4|2/4|0|2|
|D8|4/4|2/4|0|2|

All entered unique actual runs complete and pass the original gates; missing plans remain failures in all4Task coverage. D8 and N8 both add A coverage on the two minus Tasks over R8. B coverage does not improve. D8 final lineage is4 diffusion descendants,2 common-zero selections and2 NO_PLAN logical outcomes; no directly selected raw diffusion seed. Lineage is not independent causal evidence. Evidence: [tables/C_actual_status.csv](E:/v64c2/v6_4/output/preference_warmstart_20261008_01/tables/C_actual_status.csv), [tables/C_actual_quality.csv](E:/v64c2/v6_4/output/preference_warmstart_20261008_01/tables/C_actual_quality.csv), [tables/B_final_attribution.csv](E:/v64c2/v6_4/output/preference_warmstart_20261008_01/tables/B_final_attribution.csv).

A requires ΔI≤0.001rad/s **and** ΔL≤0.005m; B requires actual d≥0.030m **and** ΔL≤0.005m. D8 A passes3/4: test0_minus has ΔI=+0.003586139074753497rad/s (fail), ΔL=+0.002556685216468013m; test1_minus has ΔI=+0.000765220415545068rad/s, ΔL=−0.000346223861556805m (pass), while both plus A results alias the common zero plan. D8 B passes both comparable plus Tasks: ΔL=+1.820884/−0.815246mm, d=35.723868/34.634605mm. Both minus B comparisons are N/A because R12 is NO_PLAN; they are not passes. Thus D8 retains all6 R12 observed complete/gated outcomes but preserves quality in5/6, with2 further B N/A outcomes.

N8 has A4/4, all exactly aliasing R12 plans, but B1 pass/1 fail/2 N/A: test0_plus ΔL=+0.005268783615467010m exceeds5mm; test1_plus ΔL=+0.350021mm passes. N8 also preserves5/6 available quality comparisons, with a different tradeoff. Reusing these thresholds against N8/R8 is diagnostic only. D8 versus N8 B paths shorten3.447899/1.165267mm and lose6.219580/6.206766mm clearance while remaining above30mm; D8 is not Pareto-superior and its test0_minus A deteriorates. Evidence: [tables/D_actual_pairing.json](E:/v64c2/v6_4/output/preference_warmstart_20261008_01/tables/D_actual_pairing.json) and [tables/C_actual_quality.csv](E:/v64c2/v6_4/output/preference_warmstart_20261008_01/tables/C_actual_quality.csv).

Prediction first-admissible slots on the two minus Tasks are R12:12/12, N8:2/2, D8:2/4; on plus Tasks all methods hit at1. First-B slots on plus Tasks are R:2/2, N:2/2, D:3/2. D does not establish earlier discovery than retrieval. Four-slot prefixes are prediction-only; B censoring and R12 NO_PLAN stay visible. Evidence: [tables/B_first_hits.json](E:/v64c2/v6_4/output/preference_warmstart_20261008_01/tables/B_first_hits.json).

## Cost bounds, including inference

The frozen planner inner cold timer includes initializer/inference but omits worker startup and some input checks. Lower = recorded endpoint cold; upper = matching search worker elapsed. For R8, upper = R worker elapsed − final selection elapsed + prefix08 elapsed. These are measured scope brackets, not statistical confidence intervals.

|Task|R8 cold bracket (s)|R12 cold bracket (s)|N8 cold bracket (s)|D8 cold bracket (s)|
|---|---:|---:|---:|---:|
|c2_test0_plus|665.715380–666.168168|1000.137657–1000.590445|670.569324–671.060212|673.675942–674.285141|
|c2_test0_minus|239.938994–240.378782|469.946401–470.386189|596.834854–597.279286|573.874596–574.472161|
|c2_test1_plus|673.860115–674.318451|1008.390301–1008.848637|671.280380–671.747705|589.418890–590.014774|
|c2_test1_minus|237.904846–238.364793|409.866539–410.326487|498.420801–498.865286|497.589824–498.142704|

Conservative savings = baseline.lower − method.upper; positive means the method is cheaper within the recorded bracket. A/B share search cost, counted once per Task.

|Task|R12−D8 saving lower (s)|N8−D8 saving lower (s)|R8−D8 saving lower (s)|R12−N8 saving lower (s)|
|---|---:|---:|---:|---:|
|c2_test0_plus|325.852516|-3.715817|-8.569761|329.077445|
|c2_test0_minus|-104.525760|22.362693|-334.533168|-127.332885|
|c2_test1_plus|418.375527|81.265606|83.845341|336.642596|
|c2_test1_minus|-88.276164|0.278097|-260.237858|-88.998747|
|All4Task mean|137.856530|25.047645|-129.873861|112.347102|

D8 saves on3/4 Tasks versus N8,2/4 versus R12,1/4 versus R8. D/N all4 mean saving is25.047645–26.098309s; D/R12 mean137.856530–138.898127s. These means include the negative Tasks and do not require every Task to be cheaper. D8 costs more than R8 on average because R8 fails early on two Tasks; coverage and costs must be read together.

R/N/D streams respectively consume48/32/32 slots, make48/35/32 proposal attempts, start48/32/31 nominal rollouts, and perform482960/393120/378550 prediction main steps. The largest D/N saving, test1_plus, coincides with raw B-v2 amplitude24.695mm exceeding20mm and occupying a no-physics slot: D runs7 rollouts versus N8. This reduction cannot itself be credited as useful learned efficiency; no counterfactual rerun was performed. Warm DDIM inference per Task is0.016997–0.019367s and model import/load approximately0.853–0.882s; full cold costs already include these. Four concurrent workers share load and there are no isolated repeats; warm latency is only an import/load-subtracted decomposition. Evidence: [tables/D_planning_costs.json](E:/v64c2/v6_4/output/preference_warmstart_20261008_01/tables/D_planning_costs.json), [tables/D_endpoint_planning_costs.json](E:/v64c2/v6_4/output/preference_warmstart_20261008_01/tables/D_endpoint_planning_costs.json), [search_phase.json](E:/v64c2/v6_4/output/preference_warmstart_20261008_01/search_phase.json) and the four R `selection.json`/`prefix_08.json` pairs.

New teacher makespan1005.7178168999963s plus training26.95999850006774s gives a recorded phase wall basis1032.677815400064s. Cumulative worker service3476.4139505000785s plus training gives3503.3739490001462s; this is not makespan. Historical data acquisition and unrecorded preparation are not included, so these are limited phase-cost bases. Dividing this wall basis by the all4 D/R12 saving lower bound yields7.490960 Tasks as **arithmetic only**; the service-cost/wall-saving ratio is25.413188, not an observed task count. D/N with the entire phase wall basis similarly gives41.228540, but that charges shared teacher cost as if incremental and is not a validated break-even. N does not require Diffusion training; teacher-only N/R12 arithmetic is8.951880 Tasks. None qualifies as quality-preserving amortization because D8 loses one A band and N8 one B band. Therefore `quality_preserving_break_even_tasks=NOT_ESTABLISHED`.

The helper’s all4-bothpreferences-success-and-eachTaskpositive convention is additional and was not treated as an acceptance gate. R12 observed ability was evaluated with B N/A preserved; negative Task savings were included in all4 means. Even under that correct weaker description, each method loses one predeclared quality comparison. Current study must stop. Broader claims would require a separately authorized study with more mother groups/seeds and isolated end-to-end timing repeats; no such work is requested here. Evidence: [teacher_phase.json](E:/v64c2/v6_4/output/preference_warmstart_20261008_01/teacher_phase.json), [model/training_summary.json](E:/v64c2/v6_4/output/preference_warmstart_20261008_01/model/training_summary.json).

## Chinese interpretation

本轮工程与有限实验已完成，学习初始化已真实运行：训练4000步，按VAL选择第250步checkpoint，TEST恰好生成8个Diffusion初值，其中7个raw合法；48个新教师槽、112个TEST槽和32个逻辑actual槽均已终结。32槽中12次唯一actual、10个严格alias、10个NO_PLAN，不能把alias或A/B视图当成独立样本。结果只代表四个TEST任务、两个新母场景和一个训练种子。

同为八槽时，D8的A完整actual及原五门禁为4/4，R8为2/4，新增覆盖来自两个minus任务；N8也达到4/4。B的actual且30mm合格覆盖在R8、R12、N8、D8均为2/4，两个minus均NO_PLAN。D8相对R8有局部覆盖改善，但覆盖没有超出TRAIN-only检索。D8的六个完整逻辑结果中四个是Diffusion搜索后代、两个来自共同零初值；这不等于raw Diffusion实际成功率。

与R12比，D8的A近质量为3/4；test0_minus的ΔI=+0.003586139 rad/s超过0.001，虽ΔL=+2.556685mm仍不能通过。B在两个可比较plus任务均通过，另外两项为N/A而非通过；D8净空为35.723868/34.634605mm。N8则A为4/4、B为1项通过、1项失败、2项N/A：test0_plus的ΔL=+5.268784mm超过5mm。两方法都保留R12已观察的完整任务能力，但各丢失一项预声明近质量。

相对N8，D8在两个B成功任务的路径缩短3.447899/1.165267mm，同时净空减少6.219580/6.206766mm但仍超过30mm；这是局部偏好取舍，并非Pareto支配。D8在test0_minus的A明显差于N8。预测首个完整可接受槽在两个minus上R为12/12、N为2/2、D为2/4，D没有比检索更早；四槽数据仅为预测。

冻结规划器内计时漏掉worker启动和部分输入检查，因此采用内计时至worker外计时的实测夹逼区间。以baseline.lower−method.upper计，四任务含负值平均节省为D8对R12 +137.856530s、对N8 +25.047645s、对R8 −129.873861s。D8对N8逐任务为−3.715817/+22.362693/+81.265606/+0.278097s；最大节省任务恰有一个非法神经槽，实际仅7次预测而N8为8次，这种少跑物理不能单独算学习收益。四worker共享负载、无隔离重复，warm只为分解估计。

新教师phase makespan为1005.717817s，训练26.959999s，合计1032.677815s；累计worker服务加训练为3503.373949s，两者不可混称。仅用该已记录阶段成本与四任务经验均值做算术，D/R12比值为7.490960个任务，但D丢失一项A近质量，因此质量保持的摊销仍为NOT_ESTABLISHED；N同样丢失一项B近质量，且检索本身不需要Diffusion训练。建议learning_benefit_established_in_pilot=NOT_ESTABLISHED、默认保留C.1规则；允许报告上述局部正结果。当前有界研究必须停止，不追加实验。

