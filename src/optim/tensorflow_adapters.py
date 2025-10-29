"""
TensorFlow 优化器适配器

将模拟计算启发式优化器适配到 TensorFlow/Keras 接口
"""
from __future__ import annotations

from typing import Optional, List, Dict, Any
import numpy as np

try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

from src.optim.analog_inspired import (
    RK4Optimizer,
    DOPRI54Optimizer,
    IMEXOptimizer,
    SymplecticOptimizer,
    SDEOptimizer,
)


if not TENSORFLOW_AVAILABLE:
    raise ImportError("TensorFlow is required for tensorflow_adapters. Install with: pip install tensorflow")


class TFAnalogOptimizer(tf.keras.optimizers.Optimizer):
    """TensorFlow 优化器基类"""
    
    def __init__(self, learning_rate=1e-3, name="TFAnalogOptimizer", **kwargs):
        super().__init__(name=name, **kwargs)
        self._set_hyper("learning_rate", learning_rate)
        self.numpy_optimizer = None
        self._closure = None
        
    def _create_numpy_optimizer(self, loss_grad_fn, theta0, lr):
        """子类实现：创建对应的NumPy优化器"""
        raise NotImplementedError
    
    def _resource_apply_dense(self, grad, var, apply_state=None):
        """TensorFlow要求的接口（我们在apply_gradients中统一处理）"""
        pass
    
    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        """稀疏梯度处理（暂不支持）"""
        raise NotImplementedError("Sparse gradients not supported")
    
    def apply_gradients(self, grads_and_vars, name=None, **kwargs):
        """应用梯度（主要接口）"""
        # 提取变量和梯度
        variables = [v for g, v in grads_and_vars if g is not None]
        gradients = [g for g, v in grads_and_vars if g is not None]
        
        if not variables:
            return tf.no_op()
        
        # 转换为NumPy
        theta = np.concatenate([v.numpy().flatten() for v in variables])
        grad_np = np.concatenate([g.numpy().flatten() for g in gradients])
        
        # 初始化NumPy优化器
        if self.numpy_optimizer is None:
            lr = self._get_hyper("learning_rate")
            if isinstance(lr, tf.Variable):
                lr = lr.numpy()
            
            def loss_grad_fn(theta_in, x, y, task):
                # 简单返回预计算的梯度
                return 0.0, grad_np
            
            self.numpy_optimizer = self._create_numpy_optimizer(loss_grad_fn, theta, lr)
        
        # 更新参数
        self.numpy_optimizer.theta = theta
        theta_new, _ = self.numpy_optimizer.step()
        
        # 写回TensorFlow变量
        offset = 0
        for var in variables:
            numel = tf.size(var).numpy()
            var.assign(theta_new[offset:offset+numel].reshape(var.shape))
            offset += numel
        
        return tf.no_op()
    
    def get_config(self):
        """序列化配置"""
        config = super().get_config()
        config.update({
            "learning_rate": self._serialize_hyperparameter("learning_rate"),
        })
        return config


class TFRK4(TFAnalogOptimizer):
    """TensorFlow RK4 优化器"""
    
    def __init__(self, learning_rate=1e-3, track_energy=True, name="RK4", **kwargs):
        super().__init__(learning_rate=learning_rate, name=name, **kwargs)
        self.track_energy = track_energy
    
    def _create_numpy_optimizer(self, loss_grad_fn, theta0, lr):
        return RK4Optimizer(loss_grad_fn, theta0, lr, track_energy=self.track_energy)
    
    def get_config(self):
        config = super().get_config()
        config.update({"track_energy": self.track_energy})
        return config


class TFDOPRI54(TFAnalogOptimizer):
    """TensorFlow DOPRI54 优化器"""
    
    def __init__(
        self,
        learning_rate=1e-3,
        rtol=1e-4,
        atol=1e-7,
        track_energy=True,
        name="DOPRI54",
        **kwargs
    ):
        super().__init__(learning_rate=learning_rate, name=name, **kwargs)
        self.rtol = rtol
        self.atol = atol
        self.track_energy = track_energy
    
    def _create_numpy_optimizer(self, loss_grad_fn, theta0, lr):
        return DOPRI54Optimizer(
            loss_grad_fn, theta0, lr,
            rtol=self.rtol,
            atol=self.atol,
            track_energy=self.track_energy
        )
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "rtol": self.rtol,
            "atol": self.atol,
            "track_energy": self.track_energy
        })
        return config


class TFIMEX(TFAnalogOptimizer):
    """TensorFlow IMEX 优化器"""
    
    def __init__(
        self,
        learning_rate=1e-3,
        implicit_mass=1.0,
        damping=0.0,
        max_iter=25,
        tol=1e-6,
        track_energy=True,
        name="IMEX",
        **kwargs
    ):
        super().__init__(learning_rate=learning_rate, name=name, **kwargs)
        self.implicit_mass = implicit_mass
        self.damping = damping
        self.max_iter = max_iter
        self.tol = tol
        self.track_energy = track_energy
    
    def _create_numpy_optimizer(self, loss_grad_fn, theta0, lr):
        return IMEXOptimizer(
            loss_grad_fn, theta0, lr,
            implicit_mass=self.implicit_mass,
            damping=self.damping,
            max_iter=self.max_iter,
            tol=self.tol,
            track_energy=self.track_energy
        )
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "implicit_mass": self.implicit_mass,
            "damping": self.damping,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "track_energy": self.track_energy
        })
        return config


class TFSymplectic(TFAnalogOptimizer):
    """TensorFlow 辛积分优化器"""
    
    def __init__(
        self,
        learning_rate=1e-3,
        gamma=0.1,
        track_energy=True,
        name="Symplectic",
        **kwargs
    ):
        super().__init__(learning_rate=learning_rate, name=name, **kwargs)
        self.gamma = gamma
        self.track_energy = track_energy
    
    def _create_numpy_optimizer(self, loss_grad_fn, theta0, lr):
        return SymplecticOptimizer(
            loss_grad_fn, theta0, lr,
            gamma=self.gamma,
            track_energy=self.track_energy
        )
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "gamma": self.gamma,
            "track_energy": self.track_energy
        })
        return config


class TFSDE(TFAnalogOptimizer):
    """TensorFlow SDE 优化器"""
    
    def __init__(
        self,
        learning_rate=1e-3,
        sigma=1e-3,
        quant_bits=None,
        seed=None,
        track_energy=True,
        name="SDE",
        **kwargs
    ):
        super().__init__(learning_rate=learning_rate, name=name, **kwargs)
        self.sigma = sigma
        self.quant_bits = quant_bits
        self.seed = seed
        self.track_energy = track_energy
    
    def _create_numpy_optimizer(self, loss_grad_fn, theta0, lr):
        return SDEOptimizer(
            loss_grad_fn, theta0, lr,
            sigma=self.sigma,
            quant_bits=self.quant_bits,
            seed=self.seed,
            track_energy=self.track_energy
        )
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "sigma": self.sigma,
            "quant_bits": self.quant_bits,
            "seed": self.seed,
            "track_energy": self.track_energy
        })
        return config




