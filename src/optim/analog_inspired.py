"""
模拟计算启发式优化器 - NumPy实现

提供五种基于ODE积分器的优化器：
1. RK4Optimizer - 四阶龙格-库塔
2. DOPRI54Optimizer - 自适应Dormand-Prince
3. IMEXOptimizer - 半隐式方法（刚性问题）
4. SymplecticOptimizer - 辛积分（动量保持）
5. SDEOptimizer - 随机微分方程（噪声鲁棒）
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Dict, Any
import numpy as np

from src.ode.integrators import NFECounter, rk4_step, dopri54_step
from src.ode.implicit import imex_step, LinearSolveStats
from src.ode.symplectic import damped_symplectic_heavy_ball_step
from src.ode.sde import euler_maruyama_step
from src.metrics.energy_proxy import EnergyProxy


LossGradFn = Callable[[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], str], Tuple[float, np.ndarray]]


@dataclass
class OptimizerState:
    """优化器状态"""
    step_count: int = 0
    nfe_counter: NFECounter = None
    energy_proxy: EnergyProxy = None
    # 特定优化器的额外状态
    velocity: Optional[np.ndarray] = None  # for Symplectic
    dt_adaptive: float = 1e-3  # for DOPRI54
    
    def __post_init__(self):
        if self.nfe_counter is None:
            self.nfe_counter = NFECounter()
        if self.energy_proxy is None:
            self.energy_proxy = EnergyProxy()


class AnalogInspiredOptimizer(ABC):
    """模拟计算启发式优化器基类"""
    
    def __init__(
        self,
        loss_and_grad_fn: LossGradFn,
        theta0: np.ndarray,
        lr: float = 1e-3,
        track_energy: bool = True,
    ):
        """
        Args:
            loss_and_grad_fn: 损失和梯度计算函数 (theta, x, y, task) -> (loss, grad)
            theta0: 初始参数
            lr: 学习率（等价于ODE积分步长 dt）
            track_energy: 是否跟踪能耗指标
        """
        self.loss_and_grad_fn = loss_and_grad_fn
        self.theta = theta0.copy()
        self.lr = lr
        self.track_energy = track_energy
        self.state = OptimizerState()
        
    @abstractmethod
    def step(
        self,
        x_batch: Optional[np.ndarray] = None,
        y_batch: Optional[np.ndarray] = None,
        task: str = "regression",
    ) -> Tuple[np.ndarray, float]:
        """
        执行一步优化
        
        Returns:
            (theta_new, loss): 更新后的参数和当前损失
        """
        pass
    
    def zero_grad(self):
        """兼容PyTorch接口，NumPy版本不需要显式清零梯度"""
        pass
    
    def state_dict(self) -> Dict[str, Any]:
        """返回优化器状态字典"""
        return {
            "theta": self.theta.copy(),
            "lr": self.lr,
            "step_count": self.state.step_count,
            "nfe": self.state.nfe_counter.count,
            "total_energy": self.state.energy_proxy.energy,
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """加载优化器状态"""
        self.theta = state_dict["theta"].copy()
        self.lr = state_dict["lr"]
        self.state.step_count = state_dict["step_count"]
        self.state.nfe_counter.count = state_dict["nfe"]
    
    def get_energy_stats(self) -> Dict[str, float]:
        """获取能耗统计"""
        return {
            "total_nfe": self.state.nfe_counter.count,
            "total_energy": self.state.energy_proxy.energy,
            "avg_energy_per_step": self.state.energy_proxy.energy / max(1, self.state.step_count),
        }


class RK4Optimizer(AnalogInspiredOptimizer):
    """四阶龙格-库塔优化器
    
    基于梯度流 ODE: dθ/dt = -∇L(θ)
    使用 RK4 积分器提供 O(h^4) 局部误差
    """
    
    def step(
        self,
        x_batch: Optional[np.ndarray] = None,
        y_batch: Optional[np.ndarray] = None,
        task: str = "regression",
    ) -> Tuple[np.ndarray, float]:
        # 构建向量场：f(t, θ) = -∇L(θ)
        def vector_field(t: float, theta: np.ndarray) -> np.ndarray:
            _, grad = self.loss_and_grad_fn(theta, x_batch, y_batch, task)
            return -grad
        
        # 计算当前损失
        loss, _ = self.loss_and_grad_fn(self.theta, x_batch, y_batch, task)
        
        # RK4 步进
        t = float(self.state.step_count) * self.lr
        self.theta = rk4_step(vector_field, t, self.theta, self.lr, self.state.nfe_counter)
        
        # 更新能耗（RK4每步4次函数评估）
        if self.track_energy:
            flops_per_eval = 2 * self.theta.size  # 前向+反向
            self.state.energy_proxy.add_flops(4 * flops_per_eval)
        
        self.state.step_count += 1
        return self.theta, float(loss)


class DOPRI54Optimizer(AnalogInspiredOptimizer):
    """自适应Dormand-Prince优化器
    
    使用自适应步长控制，根据误差估计动态调整步长
    适合非刚性、平滑的损失景观
    """
    
    def __init__(
        self,
        loss_and_grad_fn: LossGradFn,
        theta0: np.ndarray,
        lr: float = 1e-3,
        rtol: float = 1e-4,
        atol: float = 1e-7,
        track_energy: bool = True,
    ):
        super().__init__(loss_and_grad_fn, theta0, lr, track_energy)
        self.rtol = rtol
        self.atol = atol
        self.state.dt_adaptive = lr
    
    def step(
        self,
        x_batch: Optional[np.ndarray] = None,
        y_batch: Optional[np.ndarray] = None,
        task: str = "regression",
    ) -> Tuple[np.ndarray, float]:
        def vector_field(t: float, theta: np.ndarray) -> np.ndarray:
            _, grad = self.loss_and_grad_fn(theta, x_batch, y_batch, task)
            return -grad
        
        loss, _ = self.loss_and_grad_fn(self.theta, x_batch, y_batch, task)
        
        t = float(self.state.step_count) * self.lr
        theta_new, _, dt_new, accepted = dopri54_step(
            vector_field, t, self.theta, self.state.dt_adaptive,
            self.rtol, self.atol, self.state.nfe_counter
        )
        
        if accepted:
            self.theta = theta_new
            self.state.step_count += 1
        
        self.state.dt_adaptive = dt_new
        
        if self.track_energy:
            # DOPRI54 每步7次函数评估
            flops_per_eval = 2 * self.theta.size
            self.state.energy_proxy.add_flops(7 * flops_per_eval)
        
        return self.theta, float(loss)


class IMEXOptimizer(AnalogInspiredOptimizer):
    """半隐式优化器（IMEX）
    
    将损失拆分为显式项和隐式项：L = f + g
    适合刚性问题（如强正则化、批归一化网络）
    """
    
    def __init__(
        self,
        loss_and_grad_fn: LossGradFn,
        theta0: np.ndarray,
        lr: float = 1e-3,
        implicit_mass: float = 1.0,
        damping: float = 0.0,
        max_iter: int = 25,
        tol: float = 1e-6,
        track_energy: bool = True,
    ):
        super().__init__(loss_and_grad_fn, theta0, lr, track_energy)
        self.implicit_mass = implicit_mass
        self.damping = damping
        self.max_iter = max_iter
        self.tol = tol
    
    def step(
        self,
        x_batch: Optional[np.ndarray] = None,
        y_batch: Optional[np.ndarray] = None,
        task: str = "regression",
    ) -> Tuple[np.ndarray, float]:
        
        theta_new, loss, stats = imex_step(
            self.loss_and_grad_fn,
            self.theta,
            x_batch,
            y_batch,
            self.lr,
            self.implicit_mass,
            self.damping,
            task,
            self.max_iter,
            self.tol,
        )
        
        self.theta = theta_new
        self.state.step_count += 1
        
        if self.track_energy:
            # 显式评估 + CG迭代
            flops_per_eval = 2 * self.theta.size
            flops_per_cg = self.theta.size * 3  # 矩阵-向量乘法
            total_flops = flops_per_eval + stats.iters * flops_per_cg
            self.state.energy_proxy.add_flops(total_flops)
        
        return self.theta, loss


class SymplecticOptimizer(AnalogInspiredOptimizer):
    """辛积分优化器
    
    基于重球ODE，保持能量几何结构
    适合动量方法、长时间训练
    """
    
    def __init__(
        self,
        loss_and_grad_fn: LossGradFn,
        theta0: np.ndarray,
        lr: float = 1e-3,
        gamma: float = 0.1,  # 阻尼系数
        track_energy: bool = True,
    ):
        super().__init__(loss_and_grad_fn, theta0, lr, track_energy)
        self.gamma = gamma
        self.state.velocity = np.zeros_like(theta0)
    
    def step(
        self,
        x_batch: Optional[np.ndarray] = None,
        y_batch: Optional[np.ndarray] = None,
        task: str = "regression",
    ) -> Tuple[np.ndarray, float]:
        
        theta_new, v_new, loss0, loss1 = damped_symplectic_heavy_ball_step(
            self.loss_and_grad_fn,
            self.theta,
            self.state.velocity,
            x_batch,
            y_batch,
            self.lr,
            self.gamma,
            task,
        )
        
        self.theta = theta_new
        self.state.velocity = v_new
        self.state.step_count += 1
        
        if self.track_energy:
            # 辛积分每步2次梯度评估
            flops_per_eval = 2 * self.theta.size
            self.state.energy_proxy.add_flops(2 * flops_per_eval)
        
        return self.theta, loss1


class SDEOptimizer(AnalogInspiredOptimizer):
    """随机微分方程优化器
    
    基于 SDE: dθ = -∇L dt + σ dB_t
    模拟模拟电路的噪声，提供鲁棒性和正则化
    """
    
    def __init__(
        self,
        loss_and_grad_fn: LossGradFn,
        theta0: np.ndarray,
        lr: float = 1e-3,
        sigma: float = 1e-3,  # 噪声强度
        quant_bits: Optional[int] = None,  # 量化位数
        seed: Optional[int] = None,
        track_energy: bool = True,
    ):
        super().__init__(loss_and_grad_fn, theta0, lr, track_energy)
        self.sigma = sigma
        self.quant_bits = quant_bits
        self.rng = np.random.default_rng(seed)
    
    def step(
        self,
        x_batch: Optional[np.ndarray] = None,
        y_batch: Optional[np.ndarray] = None,
        task: str = "regression",
    ) -> Tuple[np.ndarray, float]:
        
        theta_new, loss = euler_maruyama_step(
            self.loss_and_grad_fn,
            self.theta,
            x_batch,
            y_batch,
            self.lr,
            self.sigma,
            task,
            self.rng,
            self.quant_bits,
        )
        
        self.theta = theta_new
        self.state.step_count += 1
        
        if self.track_energy:
            # Euler-Maruyama 每步1次梯度评估 + 噪声生成
            flops_per_eval = 2 * self.theta.size
            flops_noise = self.theta.size  # 随机数生成
            self.state.energy_proxy.add_flops(flops_per_eval + flops_noise)
        
        return self.theta, loss


# 便捷工厂函数
def create_optimizer(
    method: str,
    loss_and_grad_fn: LossGradFn,
    theta0: np.ndarray,
    lr: float = 1e-3,
    **kwargs
) -> AnalogInspiredOptimizer:
    """
    创建优化器的工厂函数
    
    Args:
        method: 'rk4', 'dopri54', 'imex', 'symplectic', 'sde'
        loss_and_grad_fn: 损失梯度函数
        theta0: 初始参数
        lr: 学习率
        **kwargs: 特定优化器的额外参数
    """
    method = method.lower()
    
    if method == "rk4":
        return RK4Optimizer(loss_and_grad_fn, theta0, lr, **kwargs)
    elif method == "dopri54":
        return DOPRI54Optimizer(loss_and_grad_fn, theta0, lr, **kwargs)
    elif method == "imex":
        return IMEXOptimizer(loss_and_grad_fn, theta0, lr, **kwargs)
    elif method == "symplectic":
        return SymplecticOptimizer(loss_and_grad_fn, theta0, lr, **kwargs)
    elif method == "sde":
        return SDEOptimizer(loss_and_grad_fn, theta0, lr, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}. Choose from: rk4, dopri54, imex, symplectic, sde")




