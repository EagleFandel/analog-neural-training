# 项目文件结构

本文档描述了模拟计算启发式神经网络训练系统的完整文件结构。

## 📁 根目录

```
ANN AI Training/
├── README.md                     # 项目主文档
├── CHANGELOG.md                  # 更新日志
├── PROJECT_SUMMARY.md            # 项目完成总结
├── PROJECT_STRUCTURE.md          # 本文件
├── QUICKSTART.md                 # 快速开始指南
├── requirements.txt              # Python依赖
├── setup.py                      # PyPI打包配置
├── pyproject.toml                # 项目元数据
├── MANIFEST.in                   # 打包清单
├── .gitignore                    # Git忽略文件
├── run_dashboard.py              # Dashboard启动脚本
└── logs/                         # 日志文件目录
    └── pdf_export.log            # PDF导出日志
```

## 📚 核心代码 (`src/`)

```
src/
├── __init__.py                   # 包初始化
│
├── models/                       # 神经网络模型
│   ├── __init__.py
│   └── mlp.py                    # 多层感知机
│
├── optim/                        # 优化器
│   ├── __init__.py
│   ├── analog_inspired.py        # 模拟启发优化器（RK4、DOPRI54、IMEX、Symplectic、SDE）
│   ├── baseline_adam.py          # Adam优化器
│   ├── baseline_gd.py            # 梯度下降
│   ├── baseline_rmsprop.py       # RMSProp优化器
│   ├── pytorch_adapters.py       # PyTorch适配器
│   └── tensorflow_adapters.py    # TensorFlow适配器
│
├── ode/                          # ODE积分器
│   ├── __init__.py
│   ├── integrators.py            # 基础积分器（RK4、DOPRI54）
│   ├── symplectic.py             # 辛积分器
│   ├── implicit.py               # IMEX积分器
│   ├── sde.py                    # 随机微分方程积分器
│   ├── vector_fields.py          # 向量场定义
│   ├── geometry.py               # 几何工具
│   └── event_trigger.py          # 事件触发器
│
├── hardware/                     # 硬件仿真
│   ├── __init__.py
│   ├── analog_simulator.py       # 模拟电路仿真器
│   ├── energy_models.py          # 能耗模型
│   └── constrained_training.py   # 约束训练器
│
├── metrics/                      # 性能指标
│   ├── __init__.py
│   ├── energy_proxy.py           # 能耗估算
│   ├── flops_estimator.py        # FLOPs计算
│   └── stability.py              # 稳定性分析
│
├── utils/                        # 工具函数
│   ├── __init__.py
│   ├── plot.py                   # 绘图工具
│   └── seed.py                   # 随机种子设置
│
├── visualization/                # 可视化
│   ├── __init__.py
│   ├── advanced_dashboard.py     # Streamlit高级Dashboard
│   └── dashboard.py              # 基础Dashboard
│
└── pdf_export/                   # PDF导出模块
    ├── __init__.py
    ├── models.py                 # 数据模型
    ├── collectors.py             # 数据收集器
    ├── reportlab_renderer.py     # ReportLab渲染器
    ├── logging_utils.py          # 日志工具
    └── cli_demo.py               # 命令行演示
```

## 🧪 实验脚本 (`experiments/`)

```
experiments/
├── __init__.py
├── benchmark_suite.py            # 完整基准测试套件
├── edge_device_demo.py           # 边缘设备场景演示
├── theoretical_analysis_demo.py  # 理论分析演示
├── sine_regression.py            # 正弦回归实验
├── mnist_subset.py               # MNIST子集训练
├── energy_analysis.py            # 能耗分析实验
├── imex_vs_rk.py                 # IMEX vs RK4对比
├── symplectic_vs_rk.py           # 辛积分 vs RK4对比
├── sde_generalization.py         # SDE泛化实验
├── stiff_quadratic.py            # 刚性二次优化
└── analyze_results.py            # 结果分析脚本
```

## 📖 示例代码 (`examples/`)

```
examples/
├── __init__.py
└── use_cases/                    # 应用案例
    ├── __init__.py
    ├── case1_iot_sensor.py       # IoT传感器在线学习
    ├── case2_mobile_finetuning.py # 手机端模型微调
    ├── case3_reinforcement_learning.py # 强化学习（倒立摆）
    └── case4_stiff_optimization.py # 刚性优化问题
```

## 📊 理论分析 (`analysis/`)

```
analysis/
├── __init__.py
├── pl_condition.py               # PL条件验证
├── lyapunov_analysis.py          # Lyapunov稳定性分析
└── energy_drift.py               # 能量漂移分析
```

## 📝 文档 (`docs/`)

```
docs/
├── user_guide.md                 # 用户指南（完整API文档）
├── theory.md                     # 理论背景
├── theoretical_analysis.md       # 理论分析工具文档
├── theoretical_analysis_summary.md # 理论分析实现总结
├── hardware_design_spec.md       # 硬件设计规范
├── pdf_export_guide.md           # PDF导出指南
└── lit_review.md                 # 文献综述
```

## 📄 报告 (`reports/`)

```
reports/
├── paper_draft.md                # 学术论文草稿
├── paper_outline.md              # 论文大纲
├── poster_outline.md             # 海报大纲
├── analog_computing_applications.md # 模拟计算应用综述
├── pdf_refactor_summary.md       # PDF重构总结
├── chinese_encoding_fixes.md     # 中文编码修复记录
├── chart_labels_fix.md           # 图表标签修复记录
└── example_report.pdf            # 示例PDF报告
```

## 📈 结果 (`results/`)

```
results/
├── benchmark_results.json        # 基准测试结果
├── benchmark_table.tex           # LaTeX表格
├── edge_device_iot_sensor.json   # IoT场景结果
├── imex_vs_rk.csv                # IMEX对比结果
├── mnist_ng_vs_adam.csv          # MNIST对比结果
├── mnist_ng_vs_adam_metrics.json # MNIST指标
├── quadratic_imex.csv            # 二次函数IMEX结果
├── sde_sigma0.001_qnone.csv      # SDE结果
├── sine_gd.csv                   # 正弦回归GD结果
├── sine_rk4.csv                  # 正弦回归RK4结果
├── summary_energy.csv            # 能耗总结
├── symplectic_vs_rk.csv          # 辛积分对比结果
└── figures/                      # 图表文件
    ├── .gitkeep
    └── loss.png                  # 损失曲线图
```

## ⚙️ 配置 (`config/`)

```
config/
└── logging/
    └── pdf_logging.yaml          # PDF导出日志配置
```

## 🛠️ 脚本 (`scripts/`)

```
scripts/
└── install_pdf_deps.py           # PDF依赖安装脚本
```

## 🏗️ CI/CD (`.github/`)

```
.github/
└── workflows/
    └── ci.yml                    # GitHub Actions CI配置
```

## 📦 PyPI打包文件

- `setup.py` - setuptools配置
- `pyproject.toml` - PEP 517/518项目元数据
- `MANIFEST.in` - 打包清单
- `requirements.txt` - 依赖列表

## 🚫 忽略的文件

以下文件/目录已被`.gitignore`忽略：

- `__pycache__/` - Python缓存
- `.venv/` - 虚拟环境
- `logs/*.log` - 日志文件
- `test_*.py` - 测试脚本
- `debug_*.py` - 调试脚本
- `*.pdf`（除`reports/example_report.pdf`） - 临时PDF文件
- `results/*.csv` - 结果CSV文件
- `results/*.json` - 结果JSON文件
- `reports/report_*.pdf` - 生成的报告

## 📋 文件统计

| 类别 | 数量 |
|------|------|
| Python源文件 | ~60 |
| 文档文件 | ~15 |
| 配置文件 | ~5 |
| 示例脚本 | ~15 |
| **总计** | **~95** |

## 🔍 关键入口点

### 运行实验

```bash
# 基准测试
python experiments/benchmark_suite.py --all --steps 100

# 边缘设备演示
python experiments/edge_device_demo.py --scenario iot_sensor

# 理论分析
python experiments/theoretical_analysis_demo.py
```

### 启动Dashboard

```bash
# 推荐方式
python run_dashboard.py

# 或直接使用streamlit
streamlit run src/visualization/advanced_dashboard.py
```

### 生成PDF报告

```bash
# 命令行生成
python src/pdf_export/cli_demo.py

# 或在Dashboard中点击"生成PDF报告"按钮
```

### 运行应用案例

```bash
python examples/use_cases/case1_iot_sensor.py
python examples/use_cases/case2_mobile_finetuning.py
python examples/use_cases/case3_reinforcement_learning.py
python examples/use_cases/case4_stiff_optimization.py
```

## 📖 更新日志

查看 `CHANGELOG.md` 了解项目更新历史。

## 🤝 贡献指南

1. 新增实验脚本放入 `experiments/`
2. 新增应用案例放入 `examples/use_cases/`
3. 核心功能模块放入 `src/`
4. 文档更新到 `docs/`
5. 实验结果保存到 `results/`

---

**最后更新**: 2025-10-29  
**版本**: 1.0.0

