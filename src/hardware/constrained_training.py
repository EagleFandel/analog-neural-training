"""
硬件约束训练包装器

将优化器包装为硬件感知版本，支持功耗预算、精度限制等约束
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Callable, Tuple, Dict, Any
import numpy as np
import time

from src.optim.analog_inspired import AnalogInspiredOptimizer, create_optimizer
from src.hardware.analog_simulator import AnalogCircuitSimulator, AnalogCircuitConfig
from src.hardware.energy_models import HybridEnergyModel, DigitalEnergyModel, AnalogEnergyModel


@dataclass
class HardwareConstraints:
    """硬件约束配置"""
    # 能耗约束
    energy_budget_joules: Optional[float] = None  # 总能耗预算
    power_limit_watts: Optional[float] = None  # 功耗上限
    
    # 延迟约束
    max_latency_per_step_ms: Optional[float] = None  # 每步最大延迟
    
    # 精度约束
    min_effective_bits: Optional[int] = None  # 最小有效位数
    
    # 内存约束
    max_param_memory_mb: Optional[float] = None  # 参数内存上限


class ConstrainedTrainer:
    """硬件约束感知训练器
    
    包装优化器，在满足硬件约束的前提下进行训练
    """
    
    def __init__(
        self,
        optimizer: AnalogInspiredOptimizer,
        constraints: HardwareConstraints,
        circuit_simulator: Optional[AnalogCircuitSimulator] = None,
        energy_model: Optional[HybridEnergyModel] = None,
        verbose: bool = True
    ):
        self.optimizer = optimizer
        self.constraints = constraints
        self.simulator = circuit_simulator or AnalogCircuitSimulator()
        self.energy_model = energy_model or HybridEnergyModel()
        self.verbose = verbose
        
        # 运行统计
        self.total_energy_consumed = 0.0
        self.total_time_elapsed = 0.0
        self.steps_completed = 0
        self.constraint_violations = []
    
    def step(
        self,
        x_batch: Optional[np.ndarray] = None,
        y_batch: Optional[np.ndarray] = None,
        task: str = "regression"
    ) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        """执行一步训练（带约束检查）
        
        Returns:
            (theta, loss, stats): 参数、损失、统计信息
        """
        # 检查预算是否已用完
        if not self.can_continue():
            return self.optimizer.theta, float('inf'), {
                "stopped": True,
                "reason": "Budget exhausted or constraint violated"
            }
        
        # 记录开始时间
        start_time = time.perf_counter()
        
        # 执行优化步
        theta_before = self.optimizer.theta.copy()
        theta, loss = self.optimizer.step(x_batch, y_batch, task)
        
        # 模拟硬件效应
        if self.simulator is not None:
            theta = self.simulator.simulate_analog_gradient_computation(
                theta_before,
                theta - theta_before,  # 梯度
                self.optimizer.lr
            )
            self.optimizer.theta = theta
        
        # 计算能耗
        step_energy = self._estimate_step_energy(theta.size, x_batch.shape[0] if x_batch is not None else 1)
        self.total_energy_consumed += step_energy
        
        # 记录时间
        step_time = time.perf_counter() - start_time
        self.total_time_elapsed += step_time
        self.steps_completed += 1
        
        # 检查约束
        violations = self._check_constraints(step_energy, step_time)
        if violations:
            self.constraint_violations.extend(violations)
        
        # 统计信息
        stats = {
            "step": self.steps_completed,
            "loss": loss,
            "step_energy_j": step_energy,
            "step_time_s": step_time,
            "total_energy_j": self.total_energy_consumed,
            "total_time_s": self.total_time_elapsed,
            "violations": violations,
            "budget_remaining": self._get_remaining_budget(),
        }
        
        if self.verbose and violations:
            print(f"⚠ 步骤 {self.steps_completed}: 约束违反 - {violations}")
        
        return theta, loss, stats
    
    def can_continue(self) -> bool:
        """检查是否可以继续训练"""
        constraints = self.constraints
        
        # 能耗预算
        if constraints.energy_budget_joules is not None:
            if self.total_energy_consumed >= constraints.energy_budget_joules:
                return False
        
        # 参数内存
        if constraints.max_param_memory_mb is not None:
            param_memory_mb = self.optimizer.theta.nbytes / 1024 / 1024
            if param_memory_mb > constraints.max_param_memory_mb:
                return False
        
        return True
    
    def _estimate_step_energy(self, num_params: int, batch_size: int) -> float:
        """估算单步能耗"""
        energy_stats = self.energy_model.compute_training_step_energy(
            num_params, batch_size
        )
        return energy_stats["total"]
    
    def _check_constraints(
        self,
        step_energy: float,
        step_time: float
    ) -> list:
        """检查约束违反"""
        violations = []
        constraints = self.constraints
        
        # 功耗限制
        if constraints.power_limit_watts is not None:
            step_power = step_energy / max(step_time, 1e-9)
            if step_power > constraints.power_limit_watts:
                violations.append(
                    f"功耗超限: {step_power:.3f}W > {constraints.power_limit_watts}W"
                )
        
        # 延迟限制
        if constraints.max_latency_per_step_ms is not None:
            step_latency_ms = step_time * 1000
            if step_latency_ms > constraints.max_latency_per_step_ms:
                violations.append(
                    f"延迟超限: {step_latency_ms:.1f}ms > {constraints.max_latency_per_step_ms}ms"
                )
        
        # 精度限制
        if constraints.min_effective_bits is not None and self.simulator is not None:
            effective_bits = self.simulator.estimate_effective_bits()
            if effective_bits < constraints.min_effective_bits:
                violations.append(
                    f"精度不足: {effective_bits:.1f} bits < {constraints.min_effective_bits} bits"
                )
        
        return violations
    
    def _get_remaining_budget(self) -> Dict[str, float]:
        """获取剩余预算"""
        remaining = {}
        
        if self.constraints.energy_budget_joules is not None:
            remaining["energy_joules"] = max(
                0, self.constraints.energy_budget_joules - self.total_energy_consumed
            )
            remaining["energy_percent"] = (
                remaining["energy_joules"] / self.constraints.energy_budget_joules * 100
            )
        
        return remaining
    
    def get_summary(self) -> Dict[str, Any]:
        """获取训练摘要"""
        return {
            "steps_completed": self.steps_completed,
            "total_energy_j": self.total_energy_consumed,
            "total_time_s": self.total_time_elapsed,
            "avg_energy_per_step_j": (
                self.total_energy_consumed / self.steps_completed
                if self.steps_completed > 0 else 0
            ),
            "avg_time_per_step_s": (
                self.total_time_elapsed / self.steps_completed
                if self.steps_completed > 0 else 0
            ),
            "constraint_violations": self.constraint_violations,
            "final_theta": self.optimizer.theta,
        }


def auto_select_optimizer(
    loss_and_grad_fn: Callable,
    theta0: np.ndarray,
    lr: float = 1e-3,
    constraints: Optional[HardwareConstraints] = None,
    estimate_stiffness: bool = True
) -> AnalogInspiredOptimizer:
    """自动选择最优积分器
    
    基于损失景观刚性和硬件约束自动选择
    
    Args:
        loss_and_grad_fn: 损失梯度函数
        theta0: 初始参数
        lr: 学习率
        constraints: 硬件约束
        estimate_stiffness: 是否估计刚性
    
    Returns:
        选择的优化器
    """
    # 估计刚性比（Hessian条件数的粗略估计）
    if estimate_stiffness:
        stiffness = _estimate_loss_stiffness(loss_and_grad_fn, theta0)
    else:
        stiffness = 1.0
    
    # 根据刚性和约束选择
    if stiffness > 100:
        # 刚性问题 → IMEX
        method = "imex"
        if constraints and constraints.verbose:
            print(f"检测到刚性问题 (刚性比={stiffness:.1f})，选择 IMEX 优化器")
    
    elif constraints and constraints.energy_budget_joules is not None:
        # 能耗受限 → 自适应DOPRI54（最小化NFE）
        method = "dopri54"
        if hasattr(constraints, 'verbose') and constraints.verbose:
            print(f"能耗预算受限，选择 DOPRI54 自适应优化器")
    
    elif constraints and constraints.max_latency_per_step_ms is not None:
        # 延迟受限 → RK4（固定步长，可预测）
        method = "rk4"
        if hasattr(constraints, 'verbose') and constraints.verbose:
            print(f"延迟约束，选择 RK4 固定步长优化器")
    
    else:
        # 默认 → 辛积分（能量保持，长时间稳定）
        method = "symplectic"
    
    return create_optimizer(method, loss_and_grad_fn, theta0, lr)


def _estimate_loss_stiffness(
    loss_and_grad_fn: Callable,
    theta: np.ndarray,
    num_samples: int = 10
) -> float:
    """估计损失景观的刚性比
    
    通过随机方向的Hessian-向量积估计最大/最小特征值比
    """
    _, grad = loss_and_grad_fn(theta, None, None, "regression")
    
    if np.linalg.norm(grad) < 1e-10:
        return 1.0  # 已收敛，无刚性
    
    # 使用有限差分估计Hessian-向量积
    epsilon = 1e-5
    eigenvalue_estimates = []
    
    for _ in range(num_samples):
        # 随机方向
        v = np.random.randn(*theta.shape)
        v = v / (np.linalg.norm(v) + 1e-10)
        
        # Hessian-向量积近似：H*v ≈ (grad(theta + eps*v) - grad(theta)) / eps
        _, grad_perturbed = loss_and_grad_fn(theta + epsilon * v, None, None, "regression")
        hv = (grad_perturbed - grad) / epsilon
        
        # Rayleigh商：v^T H v / v^T v
        eigenvalue_est = np.dot(v.flatten(), hv.flatten())
        eigenvalue_estimates.append(abs(eigenvalue_est))
    
    if len(eigenvalue_estimates) < 2:
        return 1.0
    
    max_eig = max(eigenvalue_estimates)
    min_eig = min(eigenvalue_estimates)
    
    if min_eig < 1e-10:
        return float('inf')
    
    return max_eig / min_eig




