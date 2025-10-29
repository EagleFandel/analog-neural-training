"""
Lyapunov 函数分析

Lyapunov函数用于分析动力系统的稳定性：
- V(x) ≥ 0，且 V(x*) = 0
- dV/dt ≤ 0 (非增性)

在优化中，常用的Lyapunov函数包括：
1. 损失函数：V(x) = f(x) - f*
2. 能量函数：V(x, v) = f(x) + 0.5∥v∥²
"""
from __future__ import annotations

from typing import Callable, List, Tuple, Optional
import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt


@dataclass
class LyapunovAnalysisResult:
    """Lyapunov分析结果"""
    trajectory_steps: List[int]
    lyapunov_values: List[float]
    lyapunov_derivatives: List[float]
    is_decreasing: bool
    total_decrease: float
    violations: int
    
    def plot(self, save_path: Optional[str] = None):
        """绘制Lyapunov函数轨迹"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Lyapunov值
        ax1.plot(self.trajectory_steps, self.lyapunov_values, 'b-', linewidth=2)
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Lyapunov Value V(x)')
        ax1.set_title('Lyapunov Function Trajectory')
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # Lyapunov导数
        ax2.plot(self.trajectory_steps[:-1], self.lyapunov_derivatives, 'r-', linewidth=2)
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('dV/dt')
        ax2.set_title('Lyapunov Derivative (should be ≤ 0)')
        ax2.grid(True, alpha=0.3)
        
        # 标记违反点
        if self.violations > 0:
            violation_indices = [i for i, dv in enumerate(self.lyapunov_derivatives) if dv > 0]
            ax2.scatter(
                [self.trajectory_steps[i] for i in violation_indices],
                [self.lyapunov_derivatives[i] for i in violation_indices],
                color='red', s=50, zorder=5, label=f'Violations ({self.violations})'
            )
            ax2.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.show()
    
    def summary(self) -> str:
        """生成分析摘要"""
        status = "[√] 稳定" if self.is_decreasing else "[×] 不稳定"
        return (
            f"Lyapunov分析结果:\n"
            f"  状态: {status}\n"
            f"  迭代次数: {len(self.lyapunov_values)}\n"
            f"  总下降: {self.total_decrease:.6e}\n"
            f"  违反次数: {self.violations}\n"
            f"  最终值: {self.lyapunov_values[-1]:.6e}"
        )


class LyapunovAnalyzer:
    """Lyapunov函数分析器"""
    
    def __init__(
        self,
        lyapunov_fn: Callable[[np.ndarray], float],
        optimal_value: float = 0.0
    ):
        """
        参数:
            lyapunov_fn: Lyapunov函数 V(x)
            optimal_value: 最优点的Lyapunov值（通常为0）
        """
        self.lyapunov_fn = lyapunov_fn
        self.optimal_value = optimal_value
    
    def analyze_trajectory(
        self,
        trajectory: List[np.ndarray],
        dt: float = 1.0
    ) -> LyapunovAnalysisResult:
        """
        分析优化轨迹的Lyapunov函数
        
        参数:
            trajectory: 优化轨迹（参数历史）
            dt: 时间步长（用于计算导数）
        
        返回:
            LyapunovAnalysisResult
        """
        # 计算每个点的Lyapunov值
        lyapunov_values = []
        for x in trajectory:
            v = self.lyapunov_fn(x) - self.optimal_value
            lyapunov_values.append(max(v, 0.0))  # 确保非负
        
        # 计算Lyapunov导数 dV/dt ≈ (V(t+1) - V(t)) / dt
        lyapunov_derivatives = []
        violations = 0
        
        for i in range(len(lyapunov_values) - 1):
            dv = (lyapunov_values[i+1] - lyapunov_values[i]) / dt
            lyapunov_derivatives.append(dv)
            
            if dv > 1e-10:  # 允许小的数值误差
                violations += 1
        
        # 判断是否单调递减
        is_decreasing = (violations == 0) or (violations < len(lyapunov_derivatives) * 0.05)
        total_decrease = lyapunov_values[0] - lyapunov_values[-1]
        
        return LyapunovAnalysisResult(
            trajectory_steps=list(range(len(lyapunov_values))),
            lyapunov_values=lyapunov_values,
            lyapunov_derivatives=lyapunov_derivatives,
            is_decreasing=is_decreasing,
            total_decrease=total_decrease,
            violations=violations
        )


class EnergyLyapunovAnalyzer(LyapunovAnalyzer):
    """能量型Lyapunov分析器（用于动量方法）"""
    
    def __init__(
        self,
        loss_fn: Callable[[np.ndarray], float],
        optimal_loss: float = 0.0,
        momentum_weight: float = 0.5
    ):
        """
        参数:
            loss_fn: 损失函数 f(x)
            optimal_loss: 最优损失值
            momentum_weight: 动量项权重（能量函数中的系数）
        """
        self.loss_fn = loss_fn
        self.optimal_loss = optimal_loss
        self.momentum_weight = momentum_weight
        
        # Lyapunov函数：V(x, v) = f(x) - f* + 0.5 * weight * ∥v∥²
        def energy_lyapunov(state: np.ndarray) -> float:
            # 假设state = [x; v]拼接
            dim = len(state) // 2
            x = state[:dim]
            v = state[dim:]
            return (self.loss_fn(x) - self.optimal_loss + 
                   0.5 * self.momentum_weight * np.sum(v ** 2))
        
        super().__init__(energy_lyapunov, optimal_value=0.0)
    
    def analyze_momentum_trajectory(
        self,
        position_trajectory: List[np.ndarray],
        velocity_trajectory: List[np.ndarray],
        dt: float = 1.0
    ) -> LyapunovAnalysisResult:
        """
        分析带动量的优化轨迹
        
        参数:
            position_trajectory: 位置轨迹 x(t)
            velocity_trajectory: 速度轨迹 v(t)
            dt: 时间步长
        """
        # 拼接位置和速度
        combined_trajectory = []
        for x, v in zip(position_trajectory, velocity_trajectory):
            combined_trajectory.append(np.concatenate([x, v]))
        
        return self.analyze_trajectory(combined_trajectory, dt)


def compare_optimizers_stability(
    trajectories: dict[str, List[np.ndarray]],
    loss_fn: Callable[[np.ndarray], float],
    optimal_loss: float = 0.0,
    save_path: Optional[str] = None
):
    """
    比较多个优化器的稳定性
    
    参数:
        trajectories: {optimizer_name: trajectory} 字典
        loss_fn: 损失函数
        optimal_loss: 最优损失值
        save_path: 保存图像路径
    """
    # 对每个优化器进行分析
    results = {}
    analyzer = LyapunovAnalyzer(loss_fn, optimal_loss)
    
    for name, traj in trajectories.items():
        result = analyzer.analyze_trajectory(traj)
        results[name] = result
    
    # 绘制对比图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for idx, (name, result) in enumerate(results.items()):
        color = colors[idx % len(colors)]
        
        # Lyapunov值
        ax1.plot(
            result.trajectory_steps,
            result.lyapunov_values,
            color=color,
            linewidth=2,
            label=f"{name} (violations={result.violations})"
        )
        
        # Lyapunov导数
        ax2.plot(
            result.trajectory_steps[:-1],
            result.lyapunov_derivatives,
            color=color,
            linewidth=2,
            alpha=0.7,
            label=name
        )
    
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Lyapunov Value V(x)')
    ax1.set_title('Lyapunov Function Comparison')
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('dV/dt')
    ax2.set_title('Lyapunov Derivative Comparison')
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to {save_path}")
    else:
        plt.show()
    
    # 打印摘要
    print("\n" + "="*60)
    print("Lyapunov稳定性对比")
    print("="*60)
    for name, result in results.items():
        print(f"\n{name}:")
        print(result.summary())


__all__ = [
    "LyapunovAnalyzer",
    "EnergyLyapunovAnalyzer",
    "LyapunovAnalysisResult",
    "compare_optimizers_stability"
]

