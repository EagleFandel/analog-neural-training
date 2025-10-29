"""
Polyak-Łojasiewicz (PL) 条件验证

PL条件是一个比强凸性更弱但足以保证收敛的条件：
∥∇f(x)∥² ≥ 2μ(f(x) - f*)

其中 μ > 0 是PL常数，f* 是最优值
"""
from __future__ import annotations

from typing import Callable, Tuple, Optional, List
import numpy as np
from dataclasses import dataclass


@dataclass
class PLVerificationResult:
    """PL条件验证结果"""
    is_pl: bool
    pl_constant: Optional[float]
    min_ratio: float
    max_ratio: float
    mean_ratio: float
    sample_points: int
    violations: int
    
    def __str__(self) -> str:
        if self.is_pl:
            return (f"[√] 满足PL条件 (μ ≥ {self.pl_constant:.6f})\n"
                   f"  采样点: {self.sample_points}, 违反: {self.violations}\n"
                   f"  比率范围: [{self.min_ratio:.6f}, {self.max_ratio:.6f}]")
        else:
            return (f"[×] 不满足PL条件\n"
                   f"  采样点: {self.sample_points}, 违反: {self.violations}\n"
                   f"  比率范围: [{self.min_ratio:.6f}, {self.max_ratio:.6f}]")


class PLConditionVerifier:
    """PL条件验证器"""
    
    def __init__(
        self,
        loss_fn: Callable[[np.ndarray], float],
        grad_fn: Callable[[np.ndarray], np.ndarray],
        optimal_value: Optional[float] = None,
        tolerance: float = 1e-6
    ):
        """
        参数:
            loss_fn: 损失函数 f(x)
            grad_fn: 梯度函数 ∇f(x)
            optimal_value: 已知的最优值 f* (如果未知会估计)
            tolerance: 数值容差
        """
        self.loss_fn = loss_fn
        self.grad_fn = grad_fn
        self.optimal_value = optimal_value
        self.tolerance = tolerance
    
    def verify(
        self,
        sample_points: List[np.ndarray],
        estimate_optimal: bool = True
    ) -> PLVerificationResult:
        """
        在给定的采样点验证PL条件
        
        参数:
            sample_points: 参数空间中的采样点列表
            estimate_optimal: 如果True且optimal_value未知，会估计最优值
        
        返回:
            PLVerificationResult 对象
        """
        if len(sample_points) == 0:
            raise ValueError("需要至少一个采样点")
        
        # 估计最优值（如果需要）
        f_star = self.optimal_value
        if f_star is None and estimate_optimal:
            f_star = self._estimate_optimal_value(sample_points)
        elif f_star is None:
            f_star = 0.0  # 假设最优值为0
        
        # 计算每个点的PL比率
        ratios = []
        violations = 0
        
        for x in sample_points:
            f_x = self.loss_fn(x)
            grad_x = self.grad_fn(x)
            
            grad_norm_sq = np.sum(grad_x ** 2)
            f_gap = max(f_x - f_star, self.tolerance)
            
            # PL条件: ∥∇f(x)∥² ≥ 2μ(f(x) - f*)
            # 因此 μ ≤ ∥∇f(x)∥² / (2(f(x) - f*))
            if f_gap > self.tolerance:
                ratio = grad_norm_sq / (2 * f_gap)
                ratios.append(ratio)
                
                # 检查是否违反PL条件（梯度太小）
                if grad_norm_sq < self.tolerance and f_gap > self.tolerance:
                    violations += 1
            else:
                # 接近最优点，跳过
                pass
        
        if len(ratios) == 0:
            # 所有点都接近最优
            return PLVerificationResult(
                is_pl=True,
                pl_constant=None,
                min_ratio=0.0,
                max_ratio=0.0,
                mean_ratio=0.0,
                sample_points=len(sample_points),
                violations=0
            )
        
        # 分析比率
        min_ratio = np.min(ratios)
        max_ratio = np.max(ratios)
        mean_ratio = np.mean(ratios)
        
        # PL常数是所有点的最小比率
        pl_constant = min_ratio if min_ratio > self.tolerance else None
        is_pl = (pl_constant is not None) and (violations == 0)
        
        return PLVerificationResult(
            is_pl=is_pl,
            pl_constant=pl_constant,
            min_ratio=min_ratio,
            max_ratio=max_ratio,
            mean_ratio=mean_ratio,
            sample_points=len(sample_points),
            violations=violations
        )
    
    def verify_along_trajectory(
        self,
        trajectory: List[np.ndarray],
        window_size: int = 10
    ) -> List[Tuple[int, float]]:
        """
        沿优化轨迹验证PL常数的变化
        
        参数:
            trajectory: 优化轨迹（参数历史）
            window_size: 滑动窗口大小
        
        返回:
            [(step, pl_constant), ...] 列表
        """
        pl_constants = []
        
        for i in range(len(trajectory) - window_size + 1):
            window = trajectory[i:i+window_size]
            result = self.verify(window, estimate_optimal=True)
            
            if result.is_pl and result.pl_constant is not None:
                pl_constants.append((i, result.pl_constant))
            else:
                pl_constants.append((i, 0.0))
        
        return pl_constants
    
    def _estimate_optimal_value(self, points: List[np.ndarray]) -> float:
        """估计最优值（使用采样点中的最小值）"""
        values = [self.loss_fn(x) for x in points]
        return float(np.min(values))


def verify_quadratic_pl(
    A: np.ndarray,
    b: np.ndarray,
    sample_size: int = 100
) -> PLVerificationResult:
    """
    验证二次函数 f(x) = 0.5 * x^T A x + b^T x 的PL条件
    
    理论上，如果A正定，则PL常数 μ = λ_min(A)
    """
    # 定义损失和梯度
    def loss_fn(x: np.ndarray) -> float:
        return 0.5 * np.dot(x, np.dot(A, x)) + np.dot(b, x)
    
    def grad_fn(x: np.ndarray) -> np.ndarray:
        return np.dot(A, x) + b
    
    # 计算最优值
    try:
        x_opt = np.linalg.solve(A, -b)
        f_opt = loss_fn(x_opt)
    except np.linalg.LinAlgError:
        f_opt = None
    
    # 生成随机采样点
    dim = A.shape[0]
    sample_points = [np.random.randn(dim) for _ in range(sample_size)]
    
    # 验证
    verifier = PLConditionVerifier(loss_fn, grad_fn, optimal_value=f_opt)
    result = verifier.verify(sample_points)
    
    # 与理论值比较
    eigenvalues = np.linalg.eigvalsh(A)
    min_eigenvalue = np.min(eigenvalues)
    
    print(f"理论PL常数 (λ_min): {min_eigenvalue:.6f}")
    if result.pl_constant is not None:
        print(f"实验PL常数: {result.pl_constant:.6f}")
    else:
        print("实验PL常数: N/A")
    
    return result


__all__ = ["PLConditionVerifier", "PLVerificationResult", "verify_quadratic_pl"]

