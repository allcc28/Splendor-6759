# 接力计划：先按 E2 推进，E1 标记中断

基于当前证据，先用 E2 的 Stage A 结果推进 Phase 11 决策，同时明确记录 E1 尚未完成，避免后续复盘和汇报出现时间线混淆。

## 现状结论

1. E2 Stage A quick eval 已完成：`83.0% vs Greedy (n=200)`。
2. E1 有训练产物，但看起来在约 `120k` 左右中断，未完成到 300k。
3. 当前策略：先不被 E1 阻塞，继续走 E2 主线。

## 执行步骤

1. 先同步进度文档事实状态。
- E1 标记为 `interrupted/incomplete`。
- E2 标记为 `Stage A complete + gate pass candidate`。

2. 以 E2 进入 Stage B（1M full run）候选。
- 将 E2 作为当前唯一主候选推进。
- E1 放入 backlog，暂不阻塞。

3. E2 Stage B 完成后做 robust eval（n=1000）。
- 与 V4a/V5 的 authoritative baseline 对比。
- 未出 robust 结果前，不宣称超过 V5。

4. 仅在以下情况回补 E1。
- E2 Stage B 表现不达预期；或
- 报告需要完整 ablation 闭环证据。

## 关键证据文件

- `project/experiments/evaluation/maskable_ppo_eval/eval_maskable_20260315_202947.json`
- `project/logs/maskable_ppo_event_e1_no_gap_20260315_203357/logs/checkpoints/maskable_ppo_event_e1_100000_steps.zip`
- `project/logs/training_e1.log`
- `project/docs/development/PROGRESS.md`
- `project/docs/development/specs/phase11_event_based_experiment_plan.md`

## 验收条件

1. 文档中明确写出 E1 为中断，E2 为 Stage A 已完成。
2. E2 保持 Stage A 门槛结论：`vs Greedy >= 76%`。
3. Stage B 宣称升级前，必须有完整训练产物 + n=1000 robust report。
