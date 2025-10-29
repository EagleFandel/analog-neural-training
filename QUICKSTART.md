# 🚀 快速开始指南

5分钟快速体验模拟计算启发式神经网络训练！

## 前提条件

- Python 3.8+
- Git

## 安装（1分钟）

```bash
# 克隆仓库
git clone https://github.com/your-repo/analog-training.git
cd analog-training

# 安装依赖
pip install -r requirements.txt
```

## 第一个例子（2分钟）

创建文件 `test_analog_training.py`:

```python
from src.models.mlp import MLP
from src.optim.analog_inspired import create_optimizer
import numpy as np

# 生成简单数据
np.random.seed(42)
x_train = np.random.randn(200, 10)
y_train = (x_train.sum(axis=1) > 0).astype(int)

# 创建模型
model = MLP([10, 16, 2])

# 创建DOPRI54自适应优化器（推荐）
optimizer = create_optimizer(
    "dopri54",
    model.loss_and_grad,
    model.theta0,
    lr=1e-3
)

# 训练
print("开始训练...")
for step in range(50):
    theta, loss = optimizer.step(x_train, y_train, "classification")
    if step % 10 == 0:
        # 计算准确率
        preds = model.forward(theta, x_train, "classification")
        acc = np.mean(np.argmax(preds, axis=1) == y_train)
        print(f"步骤 {step}: 损失={loss:.4f}, 准确率={acc*100:.2f}%")

# 能耗统计
energy_stats = optimizer.get_energy_stats()
print(f"\n✅ 训练完成！")
print(f"总NFE: {energy_stats['total_nfe']}")
print(f"平均能耗/步: {energy_stats['avg_energy_per_step']:.2e}")
```

运行：

```bash
python test_analog_training.py
```

**预期输出**：

```
开始训练...
步骤 0: 损失=0.6932, 准确率=50.00%
步骤 10: 损失=0.4521, 准确率=78.50%
步骤 20: 损失=0.3012, 准确率=88.00%
步骤 30: 损失=0.2134, 准确率=93.00%
步骤 40: 损失=0.1623, 准确率=95.50%

✅ 训练完成！
总NFE: 350
平均能耗/步: 2.80e+05
```

## 探索更多（2分钟）

### 1. 对比不同优化器

```python
from src.optim.analog_inspired import RK4Optimizer, SymplecticOptimizer

# 尝试RK4
optimizer_rk4 = RK4Optimizer(model.loss_and_grad, model.theta0, lr=1e-3)

# 尝试辛积分（适合长期训练）
optimizer_symplectic = SymplecticOptimizer(
    model.loss_and_grad, model.theta0, lr=1e-3, gamma=0.1
)
```

### 2. 运行基准测试

```bash
python experiments/benchmark_suite.py --sine --steps 100
```

### 3. 启动可视化Dashboard

```bash
streamlit run src/visualization/advanced_dashboard.py
```

### 4. 边缘设备演示

```bash
python experiments/edge_device_demo.py --scenario iot_sensor
```

### 5. 应用案例

```bash
# IoT传感器
python examples/use_cases/case1_iot_sensor.py

# 手机端微调
python examples/use_cases/case2_mobile_finetuning.py

# 强化学习
python examples/use_cases/case3_reinforcement_learning.py

# 刚性优化
python examples/use_cases/case4_stiff_optimization.py
```

## 🎓 学习路径

1. **初学者**：
   - ✅ 运行上面的第一个例子
   - 📖 阅读 `docs/user_guide.md`
   - 🎮 尝试不同优化器

2. **进阶用户**：
   - 📊 运行基准测试套件
   - 🔬 探索理论分析工具（`analysis/`）
   - 🎨 使用可视化Dashboard

3. **研究者**：
   - 📝 阅读论文草稿（`reports/paper_draft.md`）
   - 🔧 查看硬件设计规范（`docs/hardware_design_spec.md`）
   - 💡 运行应用案例

4. **开发者**：
   - 🔌 集成到PyTorch/TensorFlow项目
   - 🛠 自定义损失函数
   - 📦 贡献代码

## 常见问题

**Q: 为什么RK4比Adam慢？**

A: RK4每步需要4次函数评估，但在**模拟硬件**上这些评估是并行的。软件仿真会慢，但实际芯片会快得多。

**Q: 哪个优化器最好？**

A: 取决于场景：
- 能耗受限 → DOPRI54
- 刚性问题 → IMEX
- 长期训练 → Symplectic
- 通用 → RK4

**Q: 如何与现有项目集成？**

A: 查看 `docs/user_guide.md` 的PyTorch/TensorFlow集成部分。

## 下一步

- 📚 阅读完整[用户指南](docs/user_guide.md)
- 🎯 查看[应用案例](examples/use_cases/)
- 💬 加入讨论（GitHub Issues）
- ⭐ 给项目加星支持！

---

**问题？** 查看 [FAQ](docs/user_guide.md#faq) 或提交 [Issue](https://github.com/your-repo/issues)

**享受高能效训练！** 🚀





