"""
理论分析工具演示

演示如何使用PL条件验证、Lyapunov分析和能量漂移分析
"""
import sys
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from analysis.pl_condition import PLConditionVerifier, verify_quadratic_pl
from analysis.lyapunov_analysis import LyapunovAnalyzer, compare_optimizers_stability
from analysis.energy_drift import LossKineticEnergyAnalyzer, compare_energy_conservation

# 设置随机种子
np.random.seed(42)


def demo_pl_condition():
    """演示PL条件验证"""
    print("=" * 70)
    print("1. PL条件验证演示")
    print("=" * 70)
    
    # 示例1: 强凸二次函数（满足PL条件）
    print("\n示例1: 强凸二次函数 f(x) = 0.5 * x^T A x")
    print("-" * 70)
    
    # 构造正定矩阵
    dim = 5
    A = np.random.randn(dim, dim)
    A = A.T @ A + np.eye(dim)  # 确保正定
    b = np.zeros(dim)
    
    result = verify_quadratic_pl(A, b, sample_size=50)
    print(result)
    
    # 示例2: 一般非线性函数
    print("\n\n示例2: Rosenbrock函数（不满足全局PL条件）")
    print("-" * 70)
    
    def rosenbrock(x):
        return sum(100*(x[i+1] - x[i]**2)**2 + (1 - x[i])**2 
                  for i in range(len(x)-1))
    
    def rosenbrock_grad(x):
        grad = np.zeros_like(x)
        for i in range(len(x)-1):
            grad[i] += -400*x[i]*(x[i+1] - x[i]**2) - 2*(1 - x[i])
            grad[i+1] += 200*(x[i+1] - x[i]**2)
        return grad
    
    # 在最优点附近采样
    sample_points = [np.ones(dim) + 0.1*np.random.randn(dim) for _ in range(50)]
    
    verifier = PLConditionVerifier(rosenbrock, rosenbrock_grad, optimal_value=0.0)
    result = verifier.verify(sample_points)
    print(result)


def demo_lyapunov_analysis():
    """演示Lyapunov稳定性分析"""
    print("\n\n" + "=" * 70)
    print("2. Lyapunov稳定性分析演示")
    print("=" * 70)
    
    # 简单二次函数
    dim = 10
    A = np.eye(dim) * 2
    b = np.ones(dim)
    
    def loss_fn(x):
        return 0.5 * np.dot(x, np.dot(A, x)) + np.dot(b, x)
    
    def grad_fn(x):
        return np.dot(A, x) + b
    
    # 运行梯度下降
    print("\n运行梯度下降...")
    x = np.random.randn(dim)
    trajectory = [x.copy()]
    
    lr = 0.1
    for _ in range(100):
        x = x - lr * grad_fn(x)
        trajectory.append(x.copy())
    
    # Lyapunov分析
    print("分析Lyapunov函数...")
    x_opt = np.linalg.solve(A, -b)
    f_opt = loss_fn(x_opt)
    
    analyzer = LyapunovAnalyzer(loss_fn, optimal_value=f_opt)
    result = analyzer.analyze_trajectory(trajectory)
    
    print("\n" + result.summary())
    
    # 保存图像（可选）
    # result.plot(save_path=PROJECT_ROOT / "results/figures/lyapunov_demo.png")


def demo_energy_drift():
    """演示能量漂移分析"""
    print("\n\n" + "=" * 70)
    print("3. 能量漂移分析演示")
    print("=" * 70)
    
    # 简单二次函数
    dim = 5
    A = np.eye(dim) * 2
    
    def loss_fn(x):
        return 0.5 * np.dot(x, np.dot(A, x))
    
    def grad_fn(x):
        return np.dot(A, x)
    
    # 模拟辛积分器（Heavy Ball）
    print("\n模拟重球法（类辛积分）...")
    x = np.random.randn(dim)
    v = np.zeros(dim)
    
    position_traj = [x.copy()]
    velocity_traj = [v.copy()]
    
    lr = 0.01
    gamma = 0.9  # 动量系数
    
    for _ in range(200):
        g = grad_fn(x)
        v = gamma * v - lr * g
        x = x + v
        
        position_traj.append(x.copy())
        velocity_traj.append(v.copy())
    
    # 能量漂移分析
    print("分析能量漂移...")
    analyzer = LossKineticEnergyAnalyzer(loss_fn, kinetic_weight=1.0)
    result = analyzer.analyze_trajectory(position_traj, velocity_traj)
    
    print("\n" + result.summary())
    
    # 保存图像（可选）
    # result.plot(save_path=PROJECT_ROOT / "results/figures/energy_drift_demo.png")


def demo_optimizer_comparison():
    """演示优化器对比分析"""
    print("\n\n" + "=" * 70)
    print("4. 优化器对比分析")
    print("=" * 70)
    
    # 测试函数
    dim = 10
    A = np.diag(np.linspace(1, 10, dim))
    
    def loss_fn(x):
        return 0.5 * np.dot(x, np.dot(A, x))
    
    def grad_fn(x):
        return np.dot(A, x)
    
    # 运行多个优化器
    x0 = np.random.randn(dim)
    trajectories = {}
    
    # Adam
    print("\n运行 Adam...")
    x = x0.copy()
    m = np.zeros(dim)
    v = np.zeros(dim)
    traj_adam = [x.copy()]
    
    beta1, beta2 = 0.9, 0.999
    lr = 0.1
    
    for t in range(1, 101):
        g = grad_fn(x)
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g**2
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        x = x - lr * m_hat / (np.sqrt(v_hat) + 1e-8)
        traj_adam.append(x.copy())
    
    trajectories['Adam'] = traj_adam
    
    # GD
    print("运行 GD...")
    x = x0.copy()
    traj_gd = [x.copy()]
    
    for _ in range(100):
        x = x - 0.1 * grad_fn(x)
        traj_gd.append(x.copy())
    
    trajectories['GD'] = traj_gd
    
    # Heavy Ball
    print("运行 Heavy Ball...")
    x = x0.copy()
    v = np.zeros(dim)
    traj_hb = [x.copy()]
    
    for _ in range(100):
        g = grad_fn(x)
        v = 0.9 * v - 0.01 * g
        x = x + v
        traj_hb.append(x.copy())
    
    trajectories['Heavy Ball'] = traj_hb
    
    # Lyapunov对比
    print("\nLyapunov稳定性对比:")
    print("-" * 70)
    
    f_opt = 0.0
    
    for name, traj in trajectories.items():
        analyzer = LyapunovAnalyzer(loss_fn, optimal_value=f_opt)
        result = analyzer.analyze_trajectory(traj)
        print(f"\n{name}:")
        print(f"  总下降: {result.total_decrease:.6e}")
        print(f"  违反次数: {result.violations}")
        print(f"  最终值: {result.lyapunov_values[-1]:.6e}")
    
    # 可以使用 compare_optimizers_stability 生成详细对比图
    # compare_optimizers_stability(trajectories, loss_fn, f_opt, 
    #                             save_path=PROJECT_ROOT / "results/figures/optimizer_comparison.png")


def main():
    """主函数"""
    print("\n" + "=" * 70)
    print("理论分析工具完整演示")
    print("=" * 70)
    
    try:
        demo_pl_condition()
        demo_lyapunov_analysis()
        demo_energy_drift()
        demo_optimizer_comparison()
        
        print("\n\n" + "=" * 70)
        print("演示完成！")
        print("=" * 70)
        print("\n提示：")
        print("  - 所有分析结果都可以保存为图像")
        print("  - 详细的API文档请参见各模块的docstring")
        print("  - 更多示例请参见 analysis/ 目录中的各模块")
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

