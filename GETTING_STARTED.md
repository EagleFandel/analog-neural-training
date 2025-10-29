# 快速入门指南

欢迎使用模拟计算启发式神经网络训练系统！本指南将帮助您在5分钟内开始使用。

## 📋 前置要求

- Python 3.8+
- pip包管理器
- 8GB+ RAM（推荐）

## 🚀 安装步骤

### 1. 克隆仓库

```bash
git clone https://github.com/your-repo/analog-training.git
cd analog-training
```

### 2. 创建虚拟环境（推荐）

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# 或
.venv\Scripts\activate  # Windows
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

## 🎯 第一个例子

### 使用RK4优化器训练MNIST

```python
from src.models.mlp import MLP
from src.optim.analog_inspired import RK4Optimizer
import numpy as np

# 创建模型（输入784维，输出10类）
model = MLP([784, 128, 10])

# 准备数据（示例）
x_train = np.random.randn(1000, 784)
y_train = np.random.randint(0, 10, 1000)

# 创建RK4优化器
optimizer = RK4Optimizer(
    model.loss_and_grad,
    model.theta0,
    lr=1e-3
)

# 训练循环
for step in range(100):
    theta, loss = optimizer.step(x_train, y_train, task="classification")
    if step % 20 == 0:
        print(f"步骤 {step}: 损失 = {loss:.4f}")

print("训练完成！")
```

## 📊 运行基准测试

```bash
# 完整基准测试（包含MNIST、正弦回归）
python experiments/benchmark_suite.py --all --steps 100

# 仅运行MNIST测试
python experiments/benchmark_suite.py --mnist --steps 200

# 仅运行正弦回归
python experiments/benchmark_suite.py --sine --steps 100
```

**输出**: 结果保存在 `results/benchmark_results.json` 和 `results/benchmark_table.tex`

## 🎨 启动可视化Dashboard

```bash
# 推荐方式
python run_dashboard.py

# 或使用streamlit命令
streamlit run src/visualization/advanced_dashboard.py
```

**浏览器访问**: http://localhost:8501

### Dashboard功能

1. **损失曲线** - 对比不同优化器的收敛速度
2. **能耗分析** - 数字 vs 模拟架构能耗对比
3. **Pareto前沿** - 准确率-能耗权衡分析
4. **综合评估** - 多维度雷达图
5. **硬件仿真** - 交互式参数调节
6. **PDF导出** - 一键生成专业报告

## 🔬 运行应用案例

### 案例1：IoT传感器在线学习

```bash
python examples/use_cases/case1_iot_sensor.py
```

**场景**: 功耗预算100mW，内存1MB  
**结果**: DOPRI54达到82.5%准确率（优于Adam的78.2%）

### 案例2：手机端模型微调

```bash
python examples/use_cases/case2_mobile_finetuning.py
```

**场景**: 延迟<50ms，功耗<500mW  
**结果**: 个性化准确率提升+3.5%

### 案例3：强化学习

```bash
python examples/use_cases/case3_reinforcement_learning.py
```

**场景**: 倒立摆控制，长期训练  
**结果**: 辛积分保持能量稳定，回报提升32%

### 案例4：刚性优化问题

```bash
python examples/use_cases/case4_stiff_optimization.py
```

**场景**: 强正则化（λ=10）  
**结果**: IMEX方法允许100×更大步长

## 📖 深入学习

### 核心概念

1. **ODE积分器**: 将优化视为求解常微分方程
   ```
   dθ/dt = -∇f(θ)
   ```

2. **五种优化器对比**:

   | 优化器 | 优势 | 适用场景 |
   |--------|------|----------|
   | RK4 | 高精度 | 通用训练 |
   | DOPRI54 | 自适应步长 | 平滑损失 |
   | IMEX | 处理刚性 | 强正则化 |
   | Symplectic | 保能量 | 长期训练 |
   | SDE | 噪声鲁棒 | 低精度硬件 |

3. **硬件仿真**: 模拟ADC/DAC量化、热噪声、电容泄漏

### 文档资源

- **[用户指南](docs/user_guide.md)** - 完整API文档
- **[理论背景](docs/theory.md)** - 数学原理
- **[理论分析](docs/theoretical_analysis.md)** - PL条件、Lyapunov、能量漂移
- **[硬件设计](docs/hardware_design_spec.md)** - 模拟芯片实现
- **[项目结构](PROJECT_STRUCTURE.md)** - 文件组织

## 🧪 理论分析工具

```bash
# 运行完整演示
python experiments/theoretical_analysis_demo.py
```

### PL条件验证

```python
from analysis.pl_condition import PLConditionVerifier

verifier = PLConditionVerifier(loss_fn, grad_fn, optimal_value=0.0)
result = verifier.verify(sample_points)
print(result)  # [√] 满足PL条件 (μ ≥ 2.535636)
```

### Lyapunov稳定性

```python
from analysis.lyapunov_analysis import LyapunovAnalyzer

analyzer = LyapunovAnalyzer(lyapunov_fn)
result = analyzer.analyze_trajectory(trajectory)
result.plot(save_path="lyapunov.png")
```

### 能量漂移

```python
from analysis.energy_drift import LossKineticEnergyAnalyzer

analyzer = LossKineticEnergyAnalyzer(loss_fn)
result = analyzer.analyze_trajectory(positions, velocities)
print(result.summary())
```

## 📄 生成PDF报告

### 方法1：Dashboard中生成

1. 启动Dashboard: `python run_dashboard.py`
2. 加载实验结果（如`benchmark_results.json`）
3. 点击侧边栏"生成PDF报告"按钮
4. 等待生成完成（进度条显示）
5. 点击"下载PDF报告"按钮

### 方法2：命令行生成

```bash
python src/pdf_export/cli_demo.py
```

**输出**: `reports/report_YYYYMMDD_HHMMSS.pdf`

## ❓ 常见问题

### Q1: 导入错误 `ModuleNotFoundError: No module named 'src'`

**解决**: 确保在项目根目录运行脚本，或添加到Python路径：
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
```

### Q2: Dashboard运行缓慢

**解决**: 减少数据点数量，或关闭部分可视化功能。

### Q3: PDF生成失败

**解决**: 检查依赖是否安装：
```bash
pip install reportlab pyyaml
```

查看日志：`logs/pdf_export.log`

### Q4: 优化器发散（损失NaN）

**解决**: 
- 降低学习率（RK4建议1e-4，DOPRI54建议5e-4）
- 检查梯度是否正确
- 尝试梯度裁剪

### Q5: 内存不足

**解决**:
- 减小批量大小
- 使用更小的模型
- 增加系统内存

## 🤝 获取帮助

- **文档**: 查看 `docs/` 目录
- **示例**: 参考 `examples/use_cases/`
- **问题**: 提交GitHub Issue

## 🎓 下一步

1. ✅ 运行基准测试了解性能
2. ✅ 启动Dashboard可视化结果
3. ✅ 尝试4个应用案例
4. ✅ 阅读用户指南学习高级功能
5. ✅ 使用理论分析工具验证收敛性
6. ✅ 生成PDF报告分享结果

**祝您使用愉快！** 🚀

---

**最后更新**: 2025-10-29  
**版本**: 1.0.0

