# 理论分析工具文档

本文档介绍如何使用理论分析工具来验证优化算法的理论性质。

## 目录

1. [PL条件验证](#pl条件验证)
2. [Lyapunov稳定性分析](#lyapunov稳定性分析)
3. [能量漂移分析](#能量漂移分析)
4. [API参考](#api参考)

---

## PL条件验证

### 理论背景

**Polyak-Łojasiewicz (PL) 条件** 是一个比强凸性更弱但足以保证收敛的条件：

$$
\|\nabla f(x)\|^2 \geq 2\mu(f(x) - f^*)
$$

其中：
- $\mu > 0$ 是PL常数
- $f^*$ 是最优值

PL条件保证了梯度下降算法的**线性收敛**速率。

### 使用方法

```python
from analysis.pl_condition import PLConditionVerifier

# 定义损失函数和梯度
def loss_fn(x):
    return 0.5 * np.dot(x, np.dot(A, x))

def grad_fn(x):
    return np.dot(A, x)

# 创建验证器
verifier = PLConditionVerifier(
    loss_fn=loss_fn,
    grad_fn=grad_fn,
    optimal_value=0.0  # 如果未知会自动估计
)

# 在采样点验证
sample_points = [np.random.randn(dim) for _ in range(100)]
result = verifier.verify(sample_points)

print(result)
# 输出: [√] 满足PL条件 (μ ≥ 1.234567)
```

### 验证二次函数

对于二次函数 $f(x) = \frac{1}{2}x^T A x + b^T x$，可以使用快捷函数：

```python
from analysis.pl_condition import verify_quadratic_pl

# 构造正定矩阵
A = np.random.randn(5, 5)
A = A.T @ A + np.eye(5)
b = np.zeros(5)

result = verify_quadratic_pl(A, b, sample_size=50)
```

理论上，如果 $A$ 正定，则PL常数 $\mu = \lambda_{\min}(A)$。

---

## Lyapunov稳定性分析

### 理论背景

**Lyapunov函数** 用于分析动力系统的稳定性：

1. $V(x) \geq 0$，且 $V(x^*) = 0$
2. $\frac{dV}{dt} \leq 0$ （非增性）

在优化中，常用的Lyapunov函数包括：
- **损失函数**: $V(x) = f(x) - f^*$
- **能量函数**: $V(x, v) = f(x) + \frac{1}{2}\|v\|^2$ （用于动量方法）

### 使用方法

```python
from analysis.lyapunov_analysis import LyapunovAnalyzer

# 定义Lyapunov函数（通常是损失函数）
def lyapunov_fn(x):
    return loss_fn(x) - optimal_loss

# 创建分析器
analyzer = LyapunovAnalyzer(
    lyapunov_fn=lyapunov_fn,
    optimal_value=0.0
)

# 运行优化算法并收集轨迹
trajectory = []
x = x0
for _ in range(100):
    trajectory.append(x.copy())
    x = x - lr * grad_fn(x)

# 分析轨迹
result = analyzer.analyze_trajectory(trajectory, dt=1.0)

print(result.summary())
# 输出:
# Lyapunov分析结果:
#   状态: [√] 稳定
#   迭代次数: 101
#   总下降: 7.377523e+00
#   违反次数: 0
#   最终值: 0.000000e+00

# 可视化
result.plot(save_path="lyapunov.png")
```

### 能量Lyapunov分析（动量方法）

```python
from analysis.lyapunov_analysis import EnergyLyapunovAnalyzer

# 创建能量型分析器
analyzer = EnergyLyapunovAnalyzer(
    loss_fn=loss_fn,
    optimal_loss=0.0,
    momentum_weight=0.5
)

# 分析带动量的优化轨迹
result = analyzer.analyze_momentum_trajectory(
    position_trajectory=positions,
    velocity_trajectory=velocities,
    dt=1.0
)
```

### 优化器对比

```python
from analysis.lyapunov_analysis import compare_optimizers_stability

trajectories = {
    "Adam": adam_trajectory,
    "GD": gd_trajectory,
    "RK4": rk4_trajectory,
}

compare_optimizers_stability(
    trajectories,
    loss_fn,
    optimal_loss=0.0,
    save_path="optimizer_comparison.png"
)
```

---

## 能量漂移分析

### 理论背景

在**辛积分**和保守系统中，能量漂移是衡量数值稳定性的重要指标：

- **理想情况**: 哈密顿量 $H(x,v)$ 应该保持恒定
- **能量漂移**: $|H(t) - H(0)|$

辛积分器（如Symplectic Euler）应该有更小的能量漂移。

### 使用方法

```python
from analysis.energy_drift import LossKineticEnergyAnalyzer

# 定义损失函数（势能）
def loss_fn(x):
    return 0.5 * np.dot(x, np.dot(A, x))

# 创建分析器
analyzer = LossKineticEnergyAnalyzer(
    loss_fn=loss_fn,
    kinetic_weight=0.5  # 动能权重
)

# 运行优化器并收集位置和速度轨迹
positions = []
velocities = []

x, v = x0, v0
for _ in range(200):
    positions.append(x.copy())
    velocities.append(v.copy())
    # 更新 x, v...

# 分析能量漂移
result = analyzer.analyze_trajectory(positions, velocities)

print(result.summary())
# 输出:
# 能量漂移分析:
#   初始能量: 2.235870e+00
#   最终能量: 2.617727e-11
#   总漂移: 2.235870e+00
#   最大漂移: 2.235870e+00
#   平均漂移: 2.154384e+00
#   漂移率: 1.112373e-02/step
#   相对漂移: 100.0000%

# 可视化
result.plot(save_path="energy_drift.png")
```

### 辛 vs 非辛对比

```python
from analysis.energy_drift import analyze_symplectic_vs_nonsymplectic

sym_result, nonsym_result = analyze_symplectic_vs_nonsymplectic(
    symplectic_traj=(sym_positions, sym_velocities),
    nonsymplectic_traj=(nonsym_positions, nonsym_velocities),
    loss_fn=loss_fn,
    save_path="symplectic_comparison.png"
)

print(f"能量漂移改进: {(1 - sym_result.total_drift/nonsym_result.total_drift)*100:.2f}%")
```

---

## API参考

### PL条件验证

#### `PLConditionVerifier`

**参数:**
- `loss_fn`: 损失函数 $f(x)$
- `grad_fn`: 梯度函数 $\nabla f(x)$
- `optimal_value`: 最优值 $f^*$ （可选）
- `tolerance`: 数值容差

**方法:**
- `verify(sample_points)`: 验证PL条件
- `verify_along_trajectory(trajectory, window_size)`: 沿轨迹验证

#### `PLVerificationResult`

**属性:**
- `is_pl`: 是否满足PL条件
- `pl_constant`: PL常数 $\mu$
- `min_ratio`, `max_ratio`, `mean_ratio`: 比率统计
- `sample_points`: 采样点数
- `violations`: 违反次数

---

### Lyapunov分析

#### `LyapunovAnalyzer`

**参数:**
- `lyapunov_fn`: Lyapunov函数 $V(x)$
- `optimal_value`: 最优点的Lyapunov值（默认0）

**方法:**
- `analyze_trajectory(trajectory, dt)`: 分析优化轨迹

#### `EnergyLyapunovAnalyzer`

**参数:**
- `loss_fn`: 损失函数 $f(x)$
- `optimal_loss`: 最优损失值
- `momentum_weight`: 动量项权重

**方法:**
- `analyze_momentum_trajectory(position_traj, velocity_traj, dt)`: 分析动量轨迹

#### `LyapunovAnalysisResult`

**属性:**
- `trajectory_steps`: 时间步列表
- `lyapunov_values`: Lyapunov值列表
- `lyapunov_derivatives`: Lyapunov导数列表
- `is_decreasing`: 是否单调递减
- `total_decrease`: 总下降量
- `violations`: 违反次数

**方法:**
- `plot(save_path)`: 绘制Lyapunov轨迹
- `summary()`: 生成分析摘要

---

### 能量漂移分析

#### `EnergyDriftAnalyzer`

**参数:**
- `hamiltonian_fn`: 哈密顿量函数 $H(x, v)$

**方法:**
- `analyze_trajectory(position_traj, velocity_traj)`: 分析能量漂移

#### `LossKineticEnergyAnalyzer`

**参数:**
- `loss_fn`: 损失函数 $f(x)$ （势能）
- `kinetic_weight`: 动能权重（默认0.5）

#### `EnergyDriftResult`

**属性:**
- `time_steps`: 时间步列表
- `energy_values`: 能量值列表
- `initial_energy`, `final_energy`: 初始/最终能量
- `total_drift`: 总漂移
- `max_drift`: 最大漂移
- `mean_drift`: 平均漂移
- `drift_rate`: 漂移率（每步）

**方法:**
- `plot(save_path)`: 绘制能量漂移图
- `summary()`: 生成分析摘要

---

## 完整示例

查看 `experiments/theoretical_analysis_demo.py` 获取完整的使用示例。

```bash
python experiments/theoretical_analysis_demo.py
```

---

## 相关文献

1. **PL条件**: Karimi, H., Nutini, J., & Schmidt, M. (2016). "Linear Convergence of Gradient and Proximal-Gradient Methods Under the Polyak-Łojasiewicz Condition"

2. **Lyapunov稳定性**: Su, W., Boyd, S., & Candès, E. (2016). "A Differential Equation for Modeling Nesterov's Accelerated Gradient Method"

3. **辛积分**: Hairer, E., Lubich, C., & Wanner, G. (2006). "Geometric Numerical Integration: Structure-Preserving Algorithms for Ordinary Differential Equations"

4. **优化的ODE视角**: Wibisono, A., Wilson, A. C., & Jordan, M. I. (2016). "A Variational Perspective on Accelerated Methods in Optimization"

---

## 常见问题

### Q: PL常数为什么是负数？

A: 如果PL常数为负数或非常小，说明函数可能不满足PL条件。尝试：
1. 在最优点附近采样
2. 增加采样点数量
3. 检查最优值估计是否准确

### Q: Lyapunov函数为什么会增加？

A: Lyapunov函数偶尔增加是正常的（数值误差、学习率过大）。如果频繁增加，可能是：
1. 学习率过大
2. 优化器不稳定
3. 损失函数非凸且陷入鞍点

### Q: 能量漂移多大算正常？

A: 这取决于积分器类型：
- **辛积分器**: 相对漂移 < 1%
- **RK4**: 相对漂移 < 10%
- **Euler**: 相对漂移可能很大

如果漂移过大，尝试减小学习率或使用辛积分器。

---

## 更新日志

- **2025-10-29**: 初始版本，实现PL条件验证、Lyapunov分析和能量漂移分析

