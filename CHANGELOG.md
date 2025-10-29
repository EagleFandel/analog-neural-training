# 更新日志

本文档记录项目的所有重要更改。

## [未发布]

### 新增

- ✨ **理论分析工具完整实现**
  - 🧮 **PL条件验证**（`analysis/pl_condition.py`）
    - `PLConditionVerifier`: 验证Polyak-Łojasiewicz条件
    - 支持任意损失函数和梯度函数
    - 提供二次函数快捷验证 `verify_quadratic_pl()`
    - 轨迹分析与窗口验证
  - 📈 **Lyapunov稳定性分析**（`analysis/lyapunov_analysis.py`）
    - `LyapunovAnalyzer`: 通用Lyapunov函数分析器
    - `EnergyLyapunovAnalyzer`: 动量方法专用分析器
    - 多优化器对比 `compare_optimizers_stability()`
    - 完整可视化支持（轨迹图、导数图）
  - ⚡ **能量漂移分析**（`analysis/energy_drift.py`）
    - `EnergyDriftAnalyzer`: 哈密顿量分析器
    - `LossKineticEnergyAnalyzer`: 损失+动能分析器
    - 辛积分 vs 非辛积分对比 `analyze_symplectic_vs_nonsymplectic()`
    - 能量守恒性能对比 `compare_energy_conservation()`
  - 🎓 演示脚本（`experiments/theoretical_analysis_demo.py`）

- ✨ **PDF导出功能重构完成**（基于ReportLab）
  - 📦 新增 `src/pdf_export/` 模块
  - 📊 支持导出所有5个标签页（损失曲线、能耗对比、Pareto前沿、综合评估、硬件仿真）
  - 🎨 使用ReportLab原生图表库（线图、柱状图、散点图、雷达图）
  - 🌏 完整支持中文文本（STSong-Light内置字体，支持系统字体自动检测）
  - 📝 专业PDF排版（A4页面、自动分页、样式化章节）
  - 📄 封面页包含元数据和章节目录
  - 🔧 CLI演示脚本（`src/pdf_export/cli_demo.py`）
  
### 改进

- ♻️ 移除旧的PDF导出依赖（weasyprint、kaleido）
- ⚡ PDF生成速度更快、更稳定
- 📋 数据收集器与Dashboard解耦，支持独立使用
- 📊 图表数据采用字典格式存储，便于序列化
- 🔍 详细日志记录到 `logs/pdf_export.log`
- 🚀 Dashboard集成新PDF导出按钮，支持进度提示
- 🐛 修复Windows PowerShell编码问题（移除emoji字符）

### 文档

- 📖 **新增** `docs/theoretical_analysis.md` - 完整的理论分析工具文档
  - PL条件验证使用指南
  - Lyapunov稳定性分析教程
  - 能量漂移分析方法
  - 完整API参考
  - 相关文献引用
- 📖 完整重写 `docs/pdf_export_guide.md`
  - 使用说明（Dashboard导出、命令行生成）
  - 编程接口示例
  - 技术架构详解
  - 故障排除FAQ
- 📝 更新 `README.md` 
  - 新增理论分析工具介绍
  - 新增理论分析演示命令
  - 更新完整文档链接
- 📋 更新 `PROJECT_SUMMARY.md` - 反映理论分析工具完成状态
- 📋 更新依赖列表（`requirements.txt`）

## [v0.2.0] - 2025-10-27

### 新增

#### 完整的演示系统

- 🎯 **基准测试套件** (`experiments/benchmark_suite.py`)
  - 支持多数据集（MNIST、Fashion-MNIST、正弦回归）
  - 对比5种优化器（Adam、GD、RK4、DOPRI54、Symplectic）
  - 多维度指标（准确率、损失、能耗、步数）
  
- 📱 **边缘设备演示** (`experiments/edge_device_demo.py`)
  - IoT传感器场景（功耗<100mW）
  - 移动端训练场景（延迟<50ms）
  - 实时场景（内存<5MB）
  
- 📊 **增强可视化Dashboard** (`src/visualization/advanced_dashboard.py`)
  - 5个标签页：损失曲线、能耗对比、Pareto前沿、综合评估、硬件仿真
  - 交互式硬件参数调节
  - 支持CSV和JSON数据源
  - 实时参数扫描分析

#### 硬件仿真模块

- 🔧 **模拟电路仿真器** (`src/hardware/analog_simulator.py`)
  - ADC/DAC量化噪声模型
  - 热噪声模拟
  - 电容泄漏效应
  - 完整的前向传播仿真
  
- ⚡ **能耗模型** (`src/hardware/energy_models.py`)
  - 纯数字架构能耗计算
  - 纯模拟架构能耗计算
  - 混合架构能耗计算
  - 详细的组件能耗分解
  
- 🎯 **约束训练器** (`src/hardware/constrained_training.py`)
  - 功耗预算控制
  - 延迟约束
  - 内存限制
  - 自适应步长调整

#### 优化器库

- 🚀 **PyTorch适配器** (`src/optim/pytorch_adapters.py`)
  - 支持5种模拟计算启发的优化器
  - 无缝集成PyTorch训练流程
  - 自动能耗跟踪
  
- 🤖 **TensorFlow适配器** (`src/optim/tensorflow_adapters.py`)
  - 支持TensorFlow 2.x
  - tf.GradientTape集成
  - 兼容Keras API

#### 应用案例

创建4个实际应用案例：
- `case1_iot_sensor.py` - IoT传感器异常检测
- `case2_mobile_finetuning.py` - 手机端模型微调
- `case3_reinforcement_learning.py` - 强化学习（倒立摆）
- `case4_stiff_optimization.py` - 刚性优化问题

#### 文档

- 📖 **硬件设计规范** (`docs/hardware_design_spec.md`)
- 📖 **用户指南** (`docs/user_guide.md`)
- 📖 **学术论文草稿** (`reports/paper_draft.md`)

### 改进

- 🔧 优化器学习率调整（RK4、DOPRI54、Symplectic需要更小的学习率）
- 🐛 修复MLP模型回归任务中的形状不匹配问题
- 📊 改进可视化图表美观性和可读性

### 修复

- 修复Dashboard加载JSON数组格式数据的问题
- 修复下载按钮显示逻辑
- 修复硬件仿真中的除零错误

## [v0.1.0] - 2025-10-25

### 新增

#### 核心ODE积分器

- 实现5种ODE积分方法（Euler、RK2、RK4、DOPRI54、辛积分）
- NFE计数器用于性能分析
- 事件触发机制
- SDE（随机微分方程）支持

#### 基础实验

- 正弦回归实验
- 二次函数刚性问题实验
- MNIST分类实验
- SDE泛化性实验

#### 指标工具

- 能耗代理估计（FLOPs、时间）
- 稳定性分析
- Lyapunov函数工具

### 初始发布

- 项目结构搭建
- 基础MLP模型
- 简单可视化工具
- README和基础文档

---

## 图例

- ✨ 新功能
- 🐛 Bug修复
- 📚 文档
- 🔧 配置/工具
- 🎨 UI/UX改进
- ⚡ 性能优化
- 🚀 新增API
- 📦 新增模块
- 📊 可视化
- 🎯 实验/案例
- 📖 文档
- 🔐 安全
- ♻️ 重构

