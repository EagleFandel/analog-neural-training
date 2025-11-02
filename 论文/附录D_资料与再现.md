# 附录D 内部资料与再现实践清单

## D1 内部资料索引
| 资料类型 | 文件/目录 | 核心内容 | 对应章节 |
|---|---|---|---|
| 理论背景 | `docs/theory.md`, `docs/theoretical_analysis.md` | ODE/PL/辛积分理论 | 2, 3 |
| 硬件设计 | `docs/hardware_design_spec.md` | 模拟/数字/混合架构 | 3 |
| 用户指南 | `docs/user_guide.md` | API与使用示例 | 3, 4 |
| PDF导出 | `docs/pdf_export_guide.md` | ReportLab实现 | 3, 4 |
| 论文草稿 | `reports/paper_draft.md` | 早期论文结构 | 1-6 |
| 应用报告 | `reports/analog_computing_applications.md` | 产业与应用背景 | 1, 6 |

## D2 再现实践清单
1. **环境安装**：`pip install -r requirements.txt`
2. **工程案例复现**：
   - 运行`experiments/benchmark_suite.py`生成`results/benchmark_results.json`
   - 运行`experiments/symplectic_vs_rk.py`等生成其他CSV
3. **图表生成**：
   - 参考`论文/04_案例分析.md`中的伪代码，读取`results/`数据，导出PNG到`论文/figs/`
4. **教育案例材料**：
   - 概念测验与量表见`附录B`
   - 任务脚本可参考`examples/use_cases/`
5. **Dashboard启动**：`python run_dashboard.py`

## D3 数据文件字段说明
- `results/benchmark_results.json`: `optimizer`, `steps`, `loss`, `accuracy`, `time`
- `results/summary_energy.csv`: `optimizer`, `energy`, `accuracy`, `nfe`
- `results/symplectic_vs_rk.csv`: `steps`, `energy_drift`, `method`
- `results/imex_vs_rk.csv`: `steps`, `stability_metric`, `method`








