Forked from TomaszOdrzygozdz/gym-splendor

## 快速定位（现在的顶层只保留开发与文档）
- `project/` — 你正在开发的新流水线（agents、reward shaping、AlphaZero、训练/评测脚本）。
- `docs/` — 计划、可行性报告、速度分析、提案等所有文档。
- `modules/` — 供新项目复用的原始环境与基线代码（已通过 `sitecustomize.py` 自动加入 `PYTHONPATH`）。
- `legacy/` — 老的实验脚本、数据、日志与备份，已整体打包收纳；除非回溯旧结果，否则不用关注。
- `setup.py` / `check_env.py` — 环境安装与快速检查入口。

## 目录详情
### 开发主线（project/）
遵循 `project/README.md` 中的 Phase 划分：
- `project/src/agents`: `score_based`、`event_based`、`alphazero` 等策略实现。
- `project/src/reward`: 奖励塑形逻辑。
- `project/src/mcts` & `project/src/nn`: AlphaZero 的搜索与模型。
- `project/configs`: 所有实验/训练配置。
- `project/scripts`: 训练、评测、锦标赛入口脚本。
- `project/experiments`: 按阶段归档的实验运行与产出。

### 文档（docs/）
- `plan.md` — 10 周执行计划。
- `Splendor_Feasibility_Report.md` — 可行性分析。
- `Evidence_Inference_Speed.md` — 推理/效率记录。
- 其他 PDF / DOCX 提案与概览。

### 遗留资源（legacy/）
包含历史数据、模型、旧脚本与日志（`artifacts/`、`data/`、`examples/`、`outputs/` 等）。需要时可从此目录取用，不再干扰主工作区。
