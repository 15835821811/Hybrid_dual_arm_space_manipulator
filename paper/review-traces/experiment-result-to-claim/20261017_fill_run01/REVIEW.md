# 独立补充与新合并库释放：结果到主张评审

日期2026-10-17；experiment-result-to-claim独立paper architect/reviewer。审阅ff_prior_origin_fill冻结协议、实现和qualified_origin_fill_001正式结果，并读取继承来源。仅离线读取/哈希/数组统计，未运行物理、推理或训练；仅写本目录。

- claim_supported: **yes**，仅针对C1“独立补充完成，新合并数据满足明示释放门”。
- C2: **not evaluated**，本轮没有神经收益实验。
- confidence: **high**，限实际数据、覆盖和41条执行证据。
- routing: **confirm bounded data release; proceed to separately frozen actual training and comparison**。
- full_goal_complete: **false**。
- paper_claim_audit: **unavailable**（paper/PAPER_CLAIM_AUDIT.json不存在）；论文主张provisional — no paper claim audit available。

## 独立核验结果

协议确实另立新run、seed20261801、全新提案身份，固定9个source coverage slots，最多72次提案/9次执行。合并资格明确保留旧失败，不是改写原512/128门。正式status FINISHED，原始审计PASS、problems=[]；以下计数从原始文件重新计算。

1. 15份generation记录=9 ADMITTED+6 REJECTED，无ERROR或未决。槽位171与443各第4次提案准入，其余7槽首次准入，全部符合每槽最多8次和首个合格即冻结。覆盖槽位恰为[171,211,243,271,279,351,431,443,491]，没有替换旧世界目标或复用旧失败任务。
2. 逐一核对9个新轨迹数组及完整初态文件哈希，全部匹配records，保存数值有限。9个准入generation保存的scene_transfer均含187项且全部true，complete_state_fields_preserved=true、goal_relabelled=false、source_failed_task_reused=false。不是通过丢warmstart或状态字段绕过转移门。
3. 与旧503前向合并后，独立计数512前向、512唯一新q17、128球场景，覆盖全部0–511槽位。原32固定修正/8父目标由不变旧来源继承。读取合并索引指向的所有TRAIN记录，重算唯一TRAIN初态555，索引12688 TRAIN/48 VAL。
4. 新frozen_validation.json与旧文件字节一致；NEW_VAL8未进入训练索引。9新TRAIN与NEW_VAL8的实际q17最小L∞距离为0.087255508rad，大于.02rad；正式审计另外检查世界目标近邻、来源身份及旧验证保护。本轮不能把这8个验证教师作为新增训练样本。
5. 全部索引record/arrays哈希复核匹配。按producer_run_id识别本批来源为旧origin_coverage535＋新origin_fill9=544；这544条中33 EXECUTION_VERIFIED、511 MODEL_VALID。合并索引new_in_this_run仅9条true，其余12727条false，这是索引相对于本run的标记，不是完整新增覆盖训练组标记。后续平衡采样必须按producer_run_id/明确样本清单取544，不能误取9条，也不能把32固定修正算独立初态。

## 原始执行及6次拒绝

audit_selection的9个sample_ids与实际准入记录完全一致，全部在执行前选定，失败不替补。离线读取旧32＋新9份report/native_trace.npz：全部门通过、重放通过、步进隐藏状态写0；ctrl长度与native/replay步数一致，qpos长度为步数+1。

| 范围 | 原始轨迹 | 含独立重放步数 | 最小执行间隙 | 最大hold位置误差 |
|---|---:|---:|---:|---:|
| 继承旧数据阶段 | 32 | 336000 | 0.050838820mm | 1.623425058mm |
| 本次新补充 | 9 | 94500 | 8.461479010mm | 0.005013660mm |
| 新合并审计 | 41 | 430500 | 0.050838820mm | 1.623425058mm |

因此41合计最差值不能标成新9独有。执行证据限上述41条，其中8条属于NEW_VAL；不能称12688库全体执行通过。仍是仿真中的原门验证，不宣称额外安全裕量、连续时间或硬件认证。

全部6次拒绝逐项检查：171_a0、171_a1、171_a2、443_a0、443_a1为INDEPENDENT_VALIDATION_REJECTED，唯一失败门geometry，目标门通过；443_a2为INITIAL_GEOMETRY_REJECTED，真实初态最小间隙-14.602274473mm。它在转移之前明确拒绝穿透，不是隐藏ERROR或移动障碍保住原样本。其后使用新提案身份的新初态/新场景，不重标旧任务。

## 旧结果与成本保持

独立重算171源码及90613来源文件SHA-256，全部匹配本run冻结身份。这包括原数据阶段文件；其status仍FINISHED_WITH_BLOCKED_ITEMS、ready_for_training_stage=false。旧39ERROR、66拒绝和旧账本完整保留。原始审计另核用户修改及旧分支保护。

inherited_accounting保留旧status/summary、账本哈希及历史旧001 unknown≤39，未记为零。此次新generation账本44 coupled，native/replay94500步；旧阶段成本单列继承，不应被表述为本次新成本或免费补跑。additional_audit_physics_calls=0只表示离线审计无额外仿真，并不等于数据补充零物理计算。

## what_results_support / dont_support

支持：在新的有限补充协议下补齐9个缺失障碍初态，保留旧逐条合格来源及独立验证，重新核验合并512前向/128新场景、32固定修正/8目标、41执行证据和来源保护，取得新的ready_for_training_stage=true。未发现阻断这一有限C1数据释放结论的缺口。

不支持：原失败作业追溯通过、历史零错误、全库执行通过、strict_multimodal、网络学到绕行类别、扩库改善了目标精度或泛化、Diffusion优于强基线。新合并门允许继承已解释而未删除的历史错误，同时要求本次无新ERROR/未决；这是明示的新资格，不是将原门改成通过。

C2尚未实际训练或比较，故本轮未评估。数据生成成功不能代替权重更新和神经价值证据；长期目标仍未完成，最终TEST和100k均未运行。

## missing_evidence / next_experiments

下一步可以在新合并父阶段的真实资格上另行冻结实际训练，不再要求修改旧ready=false。需明示12688 TRAIN实际消费、更新与抽样预算、旧权重/codec来源、验证不参与梯度/codec拟合/检查点选择、保存重载证明和成本继承。若平衡旧库与新增覆盖，以两个producer_run_id或冻结544-ID清单定义新增组，并报告每行消费；new_in_this_run仅9true不能用于代替该定义。保持完整数据谱系及等级，未执行行不升级。

随后采用预先冻结的强样条、条件均值、旧模型与新模型，同任务/种子/候选数和相同物理修正预算，在原DEV回归及未用于模型调参的NEW_VAL8上比较。分别报告raw、共同预处理后的目标残差、完整引导后准入、预定选择的真实执行重放，以任务为统计单位；不能以最终共同修正的饱和成功率单独证明NN价值。若使用NEW_VAL反复调参，则承认其DEV角色并另保留独立评估。

若要声称数据本身带来因果收益，再补同更新次数旧数据续训对照；这一更强主张的证据不是机制优先路线继续有限训练的效率门槛。保留所有强基线和物理门，查询/耗时仅辅助；不自动启动最终TEST、十万扩库或本报告之外的实验。
