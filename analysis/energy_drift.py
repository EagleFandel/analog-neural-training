"""
能量漂移分析

在辛积分和保守系统中，能量漂移是衡量数值稳定性的重要指标：
- 理想情况下，哈密顿量H(x,v)应该保持恒定
- 能量漂移 = |H(t) - H(0)|
"""
from __future__ import annotations

from typing import Callable, List, Tuple, Optional
import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt


@dataclass
class EnergyDriftResult:
    """能量漂移分析结果"""
    time_steps: List[int]
    energy_values: List[float]
    initial_energy: float
    final_energy: float
    total_drift: float
    max_drift: float
    mean_drift: float
    drift_rate: float  # 平均每步漂移
    
    def plot(self, save_path: Optional[str] = None):
        """绘制能量漂移图"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # 能量值
        ax1.plot(self.time_steps, self.energy_values, 'b-', linewidth=2)
        ax1.axhline(y=self.initial_energy, color='r', linestyle='--', 
                   label=f'Initial Energy: {self.initial_energy:.6f}', alpha=0.7)
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Total Energy H(x,v)')
        ax1.set_title('Energy Conservation')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 能量漂移
        drifts = [abs(e - self.initial_energy) for e in self.energy_values]
        ax2.plot(self.time_steps, drifts, 'g-', linewidth=2)
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Energy Drift |H(t) - H(0)|')
        ax2.set_title(f'Energy Drift (Total: {self.total_drift:.6e})')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Energy drift plot saved to {save_path}")
        else:
            plt.show()
    
    def summary(self) -> str:
        """生成分析摘要"""
        return (
            f"能量漂移分析:\n"
            f"  初始能量: {self.initial_energy:.6e}\n"
            f"  最终能量: {self.final_energy:.6e}\n"
            f"  总漂移: {self.total_drift:.6e}\n"
            f"  最大漂移: {self.max_drift:.6e}\n"
            f"  平均漂移: {self.mean_drift:.6e}\n"
            f"  漂移率: {self.drift_rate:.6e}/step\n"
            f"  相对漂移: {self.total_drift/abs(self.initial_energy)*100:.4f}%"
        )


class EnergyDriftAnalyzer:
    """能量漂移分析器"""
    
    def __init__(
        self,
        hamiltonian_fn: Callable[[np.ndarray, np.ndarray], float]
    ):
        """
        参数:
            hamiltonian_fn: 哈密顿量函数 H(x, v)
                           通常 H(x,v) = f(x) + 0.5*∥v∥²
        """
        self.hamiltonian_fn = hamiltonian_fn
    
    def analyze_trajectory(
        self,
        position_trajectory: List[np.ndarray],
        velocity_trajectory: List[np.ndarray]
    ) -> EnergyDriftResult:
        """
        分析位置-速度轨迹的能量漂移
        
        参数:
            position_trajectory: 位置轨迹 x(t)
            velocity_trajectory: 速度轨迹 v(t)
        
        返回:
            EnergyDriftResult
        """
        if len(position_trajectory) != len(velocity_trajectory):
            raise ValueError("位置和速度轨迹长度必须相同")
        
        # 计算每个时刻的能量
        energy_values = []
        for x, v in zip(position_trajectory, velocity_trajectory):
            energy = self.hamiltonian_fn(x, v)
            energy_values.append(energy)
        
        # 分析能量漂移
        initial_energy = energy_values[0]
        final_energy = energy_values[-1]
        
        drifts = [abs(e - initial_energy) for e in energy_values]
        total_drift = abs(final_energy - initial_energy)
        max_drift = np.max(drifts)
        mean_drift = np.mean(drifts)
        drift_rate = total_drift / len(energy_values) if len(energy_values) > 1 else 0.0
        
        return EnergyDriftResult(
            time_steps=list(range(len(energy_values))),
            energy_values=energy_values,
            initial_energy=initial_energy,
            final_energy=final_energy,
            total_drift=total_drift,
            max_drift=max_drift,
            mean_drift=mean_drift,
            drift_rate=drift_rate
        )


class LossKineticEnergyAnalyzer(EnergyDriftAnalyzer):
    """损失+动能分析器（常用于优化算法）"""
    
    def __init__(
        self,
        loss_fn: Callable[[np.ndarray], float],
        kinetic_weight: float = 0.5
    ):
        """
        参数:
            loss_fn: 损失函数 f(x)（势能）
            kinetic_weight: 动能权重（通常为0.5）
        """
        self.loss_fn = loss_fn
        self.kinetic_weight = kinetic_weight
        
        # 哈密顿量：H(x,v) = f(x) + 0.5*weight*∥v∥²
        def hamiltonian(x: np.ndarray, v: np.ndarray) -> float:
            potential = self.loss_fn(x)
            kinetic = 0.5 * self.kinetic_weight * np.sum(v ** 2)
            return potential + kinetic
        
        super().__init__(hamiltonian)


def compare_energy_conservation(
    trajectories: dict[str, Tuple[List[np.ndarray], List[np.ndarray]]],
    loss_fn: Callable[[np.ndarray], float],
    save_path: Optional[str] = None
):
    """
    比较多个优化器的能量守恒性能
    
    参数:
        trajectories: {optimizer_name: (position_traj, velocity_traj)} 字典
        loss_fn: 损失函数
        save_path: 保存图像路径
    """
    analyzer = LossKineticEnergyAnalyzer(loss_fn)
    results = {}
    
    # 分析每个优化器
    for name, (pos_traj, vel_traj) in trajectories.items():
        result = analyzer.analyze_trajectory(pos_traj, vel_traj)
        results[name] = result
    
    # 绘制对比图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for idx, (name, result) in enumerate(results.items()):
        color = colors[idx % len(colors)]
        
        # 能量值
        ax1.plot(
            result.time_steps,
            result.energy_values,
            color=color,
            linewidth=2,
            label=f"{name} (drift={result.total_drift:.2e})"
        )
        
        # 能量漂移
        drifts = [abs(e - result.initial_energy) for e in result.energy_values]
        ax2.plot(
            result.time_steps,
            drifts,
            color=color,
            linewidth=2,
            label=name
        )
    
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Total Energy H(x,v)')
    ax1.set_title('Energy Conservation Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Energy Drift |H(t) - H(0)|')
    ax2.set_title('Energy Drift Comparison')
    ax2.set_yscale('log')
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
    print("能量守恒性能对比")
    print("="*60)
    for name, result in results.items():
        print(f"\n{name}:")
        print(result.summary())
    
    return results


def analyze_symplectic_vs_nonsymplectic(
    symplectic_traj: Tuple[List[np.ndarray], List[np.ndarray]],
    nonsymplectic_traj: Tuple[List[np.ndarray], List[np.ndarray]],
    loss_fn: Callable[[np.ndarray], float],
    save_path: Optional[str] = None
) -> Tuple[EnergyDriftResult, EnergyDriftResult]:
    """
    对比辛积分与非辛积分的能量漂移
    
    理论上，辛积分应该有更小的能量漂移
    
    参数:
        symplectic_traj: 辛积分器轨迹 (positions, velocities)
        nonsymplectic_traj: 非辛积分器轨迹 (positions, velocities)
        loss_fn: 损失函数
        save_path: 保存图像路径
    
    返回:
        (symplectic_result, nonsymplectic_result)
    """
    analyzer = LossKineticEnergyAnalyzer(loss_fn)
    
    sym_result = analyzer.analyze_trajectory(*symplectic_traj)
    nonsym_result = analyzer.analyze_trajectory(*nonsymplectic_traj)
    
    # 绘制对比
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 辛积分 - 能量
    axes[0, 0].plot(sym_result.time_steps, sym_result.energy_values, 'b-', linewidth=2)
    axes[0, 0].axhline(y=sym_result.initial_energy, color='r', linestyle='--', alpha=0.7)
    axes[0, 0].set_title('Symplectic: Energy Conservation')
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Energy')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 辛积分 - 漂移
    sym_drifts = [abs(e - sym_result.initial_energy) for e in sym_result.energy_values]
    axes[1, 0].plot(sym_result.time_steps, sym_drifts, 'b-', linewidth=2)
    axes[1, 0].set_title(f'Symplectic: Drift = {sym_result.total_drift:.2e}')
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('Energy Drift')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 非辛积分 - 能量
    axes[0, 1].plot(nonsym_result.time_steps, nonsym_result.energy_values, 'r-', linewidth=2)
    axes[0, 1].axhline(y=nonsym_result.initial_energy, color='b', linestyle='--', alpha=0.7)
    axes[0, 1].set_title('Non-Symplectic: Energy Conservation')
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('Energy')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 非辛积分 - 漂移
    nonsym_drifts = [abs(e - nonsym_result.initial_energy) for e in nonsym_result.energy_values]
    axes[1, 1].plot(nonsym_result.time_steps, nonsym_drifts, 'r-', linewidth=2)
    axes[1, 1].set_title(f'Non-Symplectic: Drift = {nonsym_result.total_drift:.2e}')
    axes[1, 1].set_xlabel('Iteration')
    axes[1, 1].set_ylabel('Energy Drift')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison saved to {save_path}")
    else:
        plt.show()
    
    # 打印对比结果
    print("\n" + "="*60)
    print("辛积分 vs 非辛积分 能量漂移对比")
    print("="*60)
    print("\n辛积分:")
    print(sym_result.summary())
    print("\n非辛积分:")
    print(nonsym_result.summary())
    print(f"\n能量漂移改进: {(1 - sym_result.total_drift/nonsym_result.total_drift)*100:.2f}%")
    
    return sym_result, nonsym_result


__all__ = [
    "EnergyDriftAnalyzer",
    "LossKineticEnergyAnalyzer",
    "EnergyDriftResult",
    "compare_energy_conservation",
    "analyze_symplectic_vs_nonsymplectic"
]

