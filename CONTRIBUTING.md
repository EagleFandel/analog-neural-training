# 贡献指南

感谢您对模拟计算启发式神经网络训练系统的兴趣！我们欢迎各种形式的贡献。

## 🤝 如何贡献

### 报告Bug

1. 检查[Issues](https://github.com/your-repo/analog-training/issues)是否已存在相同问题
2. 如果没有，创建新Issue并包含：
   - 清晰的标题和描述
   - 重现步骤
   - 预期行为和实际行为
   - 系统环境（OS、Python版本、依赖版本）
   - 错误日志和截图

### 建议新功能

1. 创建Issue描述您的想法
2. 说明功能的用例和价值
3. 如果可能，提供设计草案或伪代码
4. 等待维护者反馈

### 提交代码

#### 设置开发环境

```bash
# 克隆仓库
git clone https://github.com/your-repo/analog-training.git
cd analog-training

# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# 或 .venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt

# 安装开发依赖（可选）
pip install black flake8 mypy pytest
```

#### 代码风格

1. **Python代码**:
   - 遵循PEP 8
   - 使用`black`格式化代码：`black src/ experiments/ examples/`
   - 使用类型提示（Type Hints）
   - 添加docstring（Google风格）

2. **命名规范**:
   - 类名：`PascalCase`（如`RK4Optimizer`）
   - 函数名：`snake_case`（如`create_optimizer`）
   - 常量：`UPPER_SNAKE_CASE`（如`MAX_ITERATIONS`）
   - 私有方法：`_leading_underscore`（如`_compute_step`）

3. **文档字符串示例**:
   ```python
   def create_optimizer(name: str, lr: float) -> AnalogInspiredOptimizer:
       """
       创建指定类型的优化器
       
       参数:
           name: 优化器名称（"rk4", "dopri54", "imex", "symplectic", "sde"）
           lr: 学习率
       
       返回:
           AnalogInspiredOptimizer实例
       
       异常:
           ValueError: 如果优化器名称无效
       
       示例:
           >>> optimizer = create_optimizer("rk4", lr=1e-3)
           >>> optimizer.step(x, y, task="classification")
       """
       ...
   ```

#### 提交流程

1. Fork仓库并创建分支：
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. 进行更改并提交：
   ```bash
   git add .
   git commit -m "Add: 简洁描述您的更改"
   ```

   **提交消息格式**:
   - `Add: 新增功能`
   - `Fix: 修复Bug`
   - `Update: 更新现有功能`
   - `Refactor: 代码重构`
   - `Docs: 文档更新`
   - `Test: 测试相关`

3. 推送到GitHub：
   ```bash
   git push origin feature/your-feature-name
   ```

4. 创建Pull Request

#### Pull Request检查清单

- [ ] 代码遵循项目风格
- [ ] 添加了必要的docstring
- [ ] 更新了相关文档
- [ ] 添加了测试（如果适用）
- [ ] 所有测试通过
- [ ] 无linter错误
- [ ] 提交消息清晰

## 📂 项目结构

贡献时请遵循以下组织规范：

```
src/              # 核心代码
├── models/       # 神经网络模型
├── optim/        # 优化器
├── ode/          # ODE积分器
├── hardware/     # 硬件仿真
├── metrics/      # 性能指标
├── utils/        # 工具函数
├── visualization/ # 可视化
└── pdf_export/   # PDF导出

experiments/      # 实验脚本
examples/         # 应用案例示例
analysis/         # 理论分析工具
docs/             # 文档
reports/          # 报告
results/          # 实验结果
```

### 文件放置指南

| 类型 | 位置 | 示例 |
|------|------|------|
| 新优化器 | `src/optim/` | `baseline_sgd.py` |
| 新模型 | `src/models/` | `cnn.py` |
| 实验脚本 | `experiments/` | `conv_net_benchmark.py` |
| 应用案例 | `examples/use_cases/` | `case5_audio_processing.py` |
| 分析工具 | `analysis/` | `convergence_rate.py` |
| 文档 | `docs/` | `advanced_features.md` |

## 🧪 测试

### 运行测试

```bash
# 运行所有测试（如果有）
pytest

# 运行特定测试
pytest tests/test_optimizers.py

# 检查代码风格
flake8 src/ experiments/
black --check src/ experiments/

# 类型检查
mypy src/
```

### 添加测试

为新功能添加测试：

```python
# tests/test_my_feature.py
import pytest
from src.optim.analog_inspired import RK4Optimizer

def test_rk4_optimizer_step():
    """测试RK4优化器的单步更新"""
    # 设置
    def loss_and_grad(theta, x, y, task):
        return 1.0, theta * 0.1
    
    theta0 = np.array([1.0, 2.0])
    optimizer = RK4Optimizer(loss_and_grad, theta0, lr=0.1)
    
    # 执行
    x = np.random.randn(10, 2)
    y = np.random.randint(0, 2, 10)
    theta, loss = optimizer.step(x, y, task="classification")
    
    # 验证
    assert theta.shape == theta0.shape
    assert isinstance(loss, float)
    assert loss > 0
```

## 📝 文档贡献

### 更新现有文档

1. 文档位于`docs/`目录
2. 使用Markdown格式
3. 保持简洁清晰
4. 添加代码示例
5. 更新目录（如果需要）

### 创建新文档

1. 文件名使用`snake_case.md`
2. 包含标题和简介
3. 添加到README的文档链接
4. 使用清晰的章节结构

### 文档模板

```markdown
# 文档标题

> 一句话描述文档内容

## 目录

- [章节1](#章节1)
- [章节2](#章节2)

## 章节1

内容...

### 代码示例

\`\`\`python
# 示例代码
from src.optim import RK4Optimizer
\`\`\`

## 章节2

内容...

---

**最后更新**: YYYY-MM-DD
```

## 🐛 Bug修复流程

1. 确认Bug存在
2. 定位问题代码
3. 编写复现测试（如果可能）
4. 修复Bug
5. 验证测试通过
6. 提交PR并引用Issue编号

## ✨ 功能开发流程

1. 讨论设计（通过Issue）
2. 创建分支
3. 实现功能
4. 添加测试
5. 编写文档
6. 提交PR
7. 代码审查
8. 合并

## 📋 代码审查标准

审查者会检查：

- ✅ 功能正确性
- ✅ 代码质量和可读性
- ✅ 性能影响
- ✅ 文档完整性
- ✅ 测试覆盖率
- ✅ 向后兼容性

## 🎓 学习资源

### 理解项目

- 阅读[用户指南](docs/user_guide.md)
- 阅读[理论背景](docs/theory.md)
- 运行示例代码

### 相关论文

1. **ODE视角的优化**: Wibisono et al. (2016)
2. **辛积分**: Hairer et al. (2006)
3. **PL条件**: Karimi et al. (2016)

## 🌟 贡献者

感谢所有贡献者！您的名字将出现在这里。

## 📞 联系方式

- GitHub Issues: https://github.com/your-repo/analog-training/issues
- Email: your-email@example.com

## 📄 许可证

贡献将遵循项目的MIT许可证。

---

**感谢您的贡献！** 🎉

