# AgenticArXiv-RL 实验复现报告

- 上游提交：`50b1ef1c345f79d3ca8e3ce1ecbc953a4f1d71b0`
- 评测协议：seed=45，repeat=3，后端=transformers，工具执行=offline_snapshot_replay
- GRPO 训练：47/120 steps，515.1s，峰值显存 8.75 GiB

## 核心结果

| 同协议数据集 | 样本数 | SFT 严格成功率 | GRPO 严格成功率 | 变化 |
|---|---:|---:|---:|---:|
| RL train diagnostic | 21 | 4.8% | 33.3% | +28.57 pp |
| Dev | 24 | 41.7% | 50.0% | +8.33 pp |

## 诊断

- Dev 工具准确率保持 87.5%，参数准确率从 59.7% 升至 62.5%。
- RL train diagnostic 的平均 token 从 3918.9 降至 3487.5（-431.4）。
- Dev 增益来自任务 `search_kw_agentic_rl`，不能外推为广泛泛化提升。
- 训练提前停止原因：five consecutive logging windows had zero within-group reward variance and reward 1.0

## 结论边界

> GRPO produced a real but narrow targeted improvement without observed dev regression. The dev gain is only two additional successes from one task, so it is positive transfer evidence rather than proof of broad generalization.

> No evidence of empty-output or false-FINISH exploitation. Correctly improved infeasible tasks emit an explicit inability thought and FINISH without calling a tool; completion and false-finish rates did not worsen, while tool failures and token cost decreased.

Base/SFT 的历史结果保留在 `metrics.csv` 中用于背景观察，但只有 seed、repeat、split 和评测协议完全一致的 v5 SFT/GRPO 对被用于因果式前后比较。
