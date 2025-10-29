"""
PyTorch 优化器适配器

将模拟计算启发式优化器适配到 PyTorch 接口
"""
from __future__ import annotations

from typing import Callable, Optional, List, Dict, Any
import numpy as np

try:
    import torch
    from torch.optim import Optimizer
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    # 创建虚拟基类
    class Optimizer:
        pass

from src.optim.analog_inspired import (
    RK4Optimizer,
    DOPRI54Optimizer,
    IMEXOptimizer,
    SymplecticOptimizer,
    SDEOptimizer,
)


if not PYTORCH_AVAILABLE:
    raise ImportError("PyTorch is required for pytorch_adapters. Install with: pip install torch")


class TorchAnalogOptimizer(Optimizer):
    """PyTorch 优化器基类"""
    
    def __init__(self, params, defaults):
        super().__init__(params, defaults)
        self.numpy_optimizer = None
        
    def _params_to_numpy(self) -> np.ndarray:
        """将PyTorch参数转换为NumPy数组"""
        params_list = []
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    params_list.append(p.data.cpu().numpy().flatten())
        return np.concatenate(params_list)
    
    def _numpy_to_params(self, theta: np.ndarray):
        """将NumPy数组写回PyTorch参数"""
        offset = 0
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    numel = p.numel()
                    p.data.copy_(
                        torch.from_numpy(
                            theta[offset:offset+numel].reshape(p.shape)
                        ).to(p.device)
                    )
                    offset += numel
    
    def _compute_loss_and_grad(
        self,
        theta: np.ndarray,
        x_batch: Optional[np.ndarray],
        y_batch: Optional[np.ndarray],
        task: str
    ):
        """包装损失和梯度计算"""
        # 将theta写回参数
        self._numpy_to_params(theta)
        
        # 计算损失（需要用户在closure中提供）
        if not hasattr(self, '_closure'):
            raise RuntimeError("Must call step(closure=...) with a closure function")
        
        loss = self._closure()
        
        # 提取梯度
        grad_list = []
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad and p.grad is not None:
                    grad_list.append(p.grad.cpu().numpy().flatten())
        
        grad = np.concatenate(grad_list) if grad_list else np.zeros_like(theta)
        
        return float(loss.item()), grad
    
    @torch.no_grad()
    def step(self, closure=None):
        """执行单步优化"""
        if closure is None:
            raise ValueError("Analog optimizers require a closure function")
        
        # 保存closure供梯度计算使用
        self._closure = closure
        
        # 初始化NumPy优化器（首次调用）
        if self.numpy_optimizer is None:
            theta0 = self._params_to_numpy()
            lr = self.param_groups[0]['lr']
            self.numpy_optimizer = self._create_numpy_optimizer(
                self._compute_loss_and_grad, theta0, lr
            )
        
        # 执行优化步
        _, loss = self.numpy_optimizer.step()
        
        # 写回参数
        self._numpy_to_params(self.numpy_optimizer.theta)
        
        return loss
    
    def _create_numpy_optimizer(self, loss_grad_fn, theta0, lr):
        """子类实现：创建对应的NumPy优化器"""
        raise NotImplementedError
    
    def state_dict(self) -> Dict[str, Any]:
        """保存优化器状态"""
        state = super().state_dict()
        if self.numpy_optimizer is not None:
            state['numpy_state'] = self.numpy_optimizer.state_dict()
        return state
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """加载优化器状态"""
        super().load_state_dict(state_dict)
        if 'numpy_state' in state_dict and self.numpy_optimizer is not None:
            self.numpy_optimizer.load_state_dict(state_dict['numpy_state'])


class TorchRK4(TorchAnalogOptimizer):
    """PyTorch RK4 优化器"""
    
    def __init__(self, params, lr=1e-3, track_energy=True):
        defaults = dict(lr=lr, track_energy=track_energy)
        super().__init__(params, defaults)
    
    def _create_numpy_optimizer(self, loss_grad_fn, theta0, lr):
        return RK4Optimizer(
            loss_grad_fn, theta0, lr,
            track_energy=self.param_groups[0]['track_energy']
        )


class TorchDOPRI54(TorchAnalogOptimizer):
    """PyTorch DOPRI54 自适应优化器"""
    
    def __init__(self, params, lr=1e-3, rtol=1e-4, atol=1e-7, track_energy=True):
        defaults = dict(lr=lr, rtol=rtol, atol=atol, track_energy=track_energy)
        super().__init__(params, defaults)
    
    def _create_numpy_optimizer(self, loss_grad_fn, theta0, lr):
        group = self.param_groups[0]
        return DOPRI54Optimizer(
            loss_grad_fn, theta0, lr,
            rtol=group['rtol'],
            atol=group['atol'],
            track_energy=group['track_energy']
        )


class TorchIMEX(TorchAnalogOptimizer):
    """PyTorch IMEX 优化器（刚性问题）"""
    
    def __init__(
        self,
        params,
        lr=1e-3,
        implicit_mass=1.0,
        damping=0.0,
        max_iter=25,
        tol=1e-6,
        track_energy=True
    ):
        defaults = dict(
            lr=lr,
            implicit_mass=implicit_mass,
            damping=damping,
            max_iter=max_iter,
            tol=tol,
            track_energy=track_energy
        )
        super().__init__(params, defaults)
    
    def _create_numpy_optimizer(self, loss_grad_fn, theta0, lr):
        group = self.param_groups[0]
        return IMEXOptimizer(
            loss_grad_fn, theta0, lr,
            implicit_mass=group['implicit_mass'],
            damping=group['damping'],
            max_iter=group['max_iter'],
            tol=group['tol'],
            track_energy=group['track_energy']
        )


class TorchSymplectic(TorchAnalogOptimizer):
    """PyTorch 辛积分优化器（动量方法）"""
    
    def __init__(self, params, lr=1e-3, gamma=0.1, track_energy=True):
        defaults = dict(lr=lr, gamma=gamma, track_energy=track_energy)
        super().__init__(params, defaults)
    
    def _create_numpy_optimizer(self, loss_grad_fn, theta0, lr):
        group = self.param_groups[0]
        return SymplecticOptimizer(
            loss_grad_fn, theta0, lr,
            gamma=group['gamma'],
            track_energy=group['track_energy']
        )


class TorchSDE(TorchAnalogOptimizer):
    """PyTorch SDE 优化器（噪声鲁棒）"""
    
    def __init__(
        self,
        params,
        lr=1e-3,
        sigma=1e-3,
        quant_bits=None,
        seed=None,
        track_energy=True
    ):
        defaults = dict(
            lr=lr,
            sigma=sigma,
            quant_bits=quant_bits,
            seed=seed,
            track_energy=track_energy
        )
        super().__init__(params, defaults)
    
    def _create_numpy_optimizer(self, loss_grad_fn, theta0, lr):
        group = self.param_groups[0]
        return SDEOptimizer(
            loss_grad_fn, theta0, lr,
            sigma=group['sigma'],
            quant_bits=group['quant_bits'],
            seed=group['seed'],
            track_energy=group['track_energy']
        )


# 便捷别名
RK4 = TorchRK4
DOPRI54 = TorchDOPRI54
IMEX = TorchIMEX
Symplectic = TorchSymplectic
SDE = TorchSDE




