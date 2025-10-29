"""
模拟电路仿真器

模拟真实模拟电路的噪声、量化、漂移等非理想特性
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class AnalogCircuitConfig:
    """模拟电路配置参数"""
    # ADC/DAC 量化
    adc_bits: int = 8  # 模数转换器位宽
    dac_bits: int = 8  # 数模转换器位宽
    
    # 热噪声（Johnson-Nyquist）
    thermal_noise_sigma: float = 1e-4  # 热噪声标准差
    temperature: float = 300.0  # 温度(K)
    
    # 电容泄漏
    leakage_rate: float = 1e-5  # 每秒电荷泄漏率
    
    # 运放非理想性
    opamp_offset: float = 1e-3  # 输入偏移电压
    opamp_gain_error: float = 0.01  # 增益误差（相对值）
    
    # 忆阻器特性
    memristor_write_noise: float = 1e-3  # 写入噪声
    memristor_drift: float = 1e-6  # 电导漂移
    
    # 跨导放大器
    transconductance_variation: float = 0.05  # 跨导变化（工艺偏差）


class AnalogCircuitSimulator:
    """模拟电路仿真器
    
    模拟真实模拟计算硬件的各种非理想效应
    """
    
    def __init__(
        self,
        config: Optional[AnalogCircuitConfig] = None,
        seed: Optional[int] = None
    ):
        self.config = config or AnalogCircuitConfig()
        self.rng = np.random.default_rng(seed)
        self.time_elapsed = 0.0
        
        # 工艺偏差（在初始化时确定，整个训练过程保持不变）
        self._transconductance_map = None
    
    def quantize_adc(self, signal: np.ndarray) -> np.ndarray:
        """模拟ADC量化（模数转换）
        
        将连续信号量化为有限精度数字值
        """
        if self.config.adc_bits is None or self.config.adc_bits >= 32:
            return signal  # 无量化
        
        levels = 2 ** self.config.adc_bits
        signal_min = signal.min()
        signal_max = signal.max()
        
        if signal_max <= signal_min:
            return signal
        
        # 量化
        step = (signal_max - signal_min) / (levels - 1)
        quantized = np.round((signal - signal_min) / step) * step + signal_min
        
        return quantized
    
    def quantize_dac(self, digital: np.ndarray) -> np.ndarray:
        """模拟DAC量化（数模转换）"""
        if self.config.dac_bits is None or self.config.dac_bits >= 32:
            return digital
        
        levels = 2 ** self.config.dac_bits
        digital_min = digital.min()
        digital_max = digital.max()
        
        if digital_max <= digital_min:
            return digital
        
        step = (digital_max - digital_min) / (levels - 1)
        quantized = np.round((digital - digital_min) / step) * step + digital_min
        
        return quantized
    
    def add_thermal_noise(self, signal: np.ndarray) -> np.ndarray:
        """添加热噪声（Johnson-Nyquist噪声）
        
        V_n = sqrt(4 k_B T R Δf)
        """
        if self.config.thermal_noise_sigma == 0:
            return signal
        
        noise = self.rng.normal(0, self.config.thermal_noise_sigma, signal.shape)
        return signal + noise
    
    def apply_capacitor_leakage(
        self,
        params: np.ndarray,
        dt: float
    ) -> np.ndarray:
        """电容电荷泄漏
        
        Q(t) = Q_0 * exp(-t / RC)
        近似：Q(t+dt) ≈ Q(t) * (1 - leakage_rate * dt)
        """
        if self.config.leakage_rate == 0:
            return params
        
        decay = 1.0 - self.config.leakage_rate * dt
        self.time_elapsed += dt
        
        return params * decay
    
    def apply_opamp_nonideality(self, signal: np.ndarray) -> np.ndarray:
        """运算放大器非理想性
        
        - 输入偏移电压
        - 有限增益误差
        """
        # 偏移
        signal_with_offset = signal + self.config.opamp_offset
        
        # 增益误差
        gain_actual = 1.0 + self.config.opamp_gain_error
        signal_with_gain = signal_with_offset * gain_actual
        
        return signal_with_gain
    
    def apply_memristor_noise(
        self,
        weights: np.ndarray,
        is_write: bool = False
    ) -> np.ndarray:
        """忆阻器噪声和漂移
        
        Args:
            weights: 权重（存储在忆阻器中）
            is_write: 是否为写操作
        """
        result = weights.copy()
        
        # 写入噪声（编程噪声）
        if is_write and self.config.memristor_write_noise > 0:
            write_noise = self.rng.normal(
                0,
                self.config.memristor_write_noise * np.abs(weights),
                weights.shape
            )
            result += write_noise
        
        # 电导漂移（时间相关）
        if self.config.memristor_drift > 0:
            drift = self.rng.normal(
                0,
                self.config.memristor_drift * np.sqrt(self.time_elapsed),
                weights.shape
            )
            result += drift
        
        return result
    
    def apply_transconductance_variation(
        self,
        gradients: np.ndarray
    ) -> np.ndarray:
        """跨导放大器工艺偏差
        
        不同放大器的跨导值会有工艺偏差（PVT变化）
        """
        if self.config.transconductance_variation == 0:
            return gradients
        
        # 初始化跨导映射（每个参数对应一个放大器）
        if self._transconductance_map is None:
            self._transconductance_map = 1.0 + self.rng.normal(
                0,
                self.config.transconductance_variation,
                gradients.shape
            )
        
        return gradients * self._transconductance_map
    
    def simulate_analog_gradient_computation(
        self,
        params: np.ndarray,
        gradients: np.ndarray,
        dt: float
    ) -> np.ndarray:
        """完整的模拟梯度计算流程
        
        模拟从数字参数 → 模拟电路 → 数字梯度的全过程
        """
        # 1. DAC：数字参数 → 模拟电压
        params_analog = self.quantize_dac(params)
        
        # 2. 模拟计算（梯度通过跨导放大器计算）
        grads_analog = self.apply_transconductance_variation(gradients)
        grads_analog = self.add_thermal_noise(grads_analog)
        grads_analog = self.apply_opamp_nonideality(grads_analog)
        
        # 3. ADC：模拟梯度 → 数字值
        grads_digital = self.quantize_adc(grads_analog)
        
        # 4. 参数更新（在模拟域，如电容）
        params_updated = params_analog - dt * grads_digital
        
        # 5. 电容泄漏
        params_updated = self.apply_capacitor_leakage(params_updated, dt)
        
        # 6. 如果使用忆阻器存储
        params_updated = self.apply_memristor_noise(params_updated, is_write=True)
        
        return params_updated
    
    def reset_time(self):
        """重置时间计数器（用于新的训练会话）"""
        self.time_elapsed = 0.0
    
    def get_stats(self) -> dict:
        """获取仿真统计信息"""
        return {
            "config": self.config.__dict__,
            "time_elapsed": self.time_elapsed,
            "effective_bits": self.estimate_effective_bits(),
        }
    
    def estimate_effective_bits(self) -> float:
        """估计有效位数（考虑噪声）
        
        ENOB = log2(V_ref / V_noise)
        """
        # 简化估计：基于热噪声和量化噪声
        quant_noise = 1.0 / (2 ** self.config.adc_bits) if self.config.adc_bits < 32 else 0
        total_noise = np.sqrt(
            self.config.thermal_noise_sigma ** 2 + quant_noise ** 2
        )
        
        if total_noise == 0:
            return float(self.config.adc_bits)
        
        enob = -np.log2(total_noise + 1e-10)
        return float(np.clip(enob, 1, self.config.adc_bits))


def create_realistic_config(scenario: str = "low_power") -> AnalogCircuitConfig:
    """创建预设的真实场景配置
    
    Args:
        scenario: "low_power" | "high_precision" | "harsh_environment"
    """
    if scenario == "low_power":
        # 低功耗场景：低精度、高噪声
        return AnalogCircuitConfig(
            adc_bits=6,
            dac_bits=6,
            thermal_noise_sigma=1e-3,
            leakage_rate=1e-4,
            opamp_offset=5e-3,
            memristor_write_noise=5e-3,
        )
    
    elif scenario == "high_precision":
        # 高精度场景：高位宽、低噪声
        return AnalogCircuitConfig(
            adc_bits=12,
            dac_bits=12,
            thermal_noise_sigma=1e-5,
            leakage_rate=1e-6,
            opamp_offset=1e-4,
            memristor_write_noise=1e-4,
        )
    
    elif scenario == "harsh_environment":
        # 恶劣环境：高温、高漂移
        return AnalogCircuitConfig(
            adc_bits=8,
            dac_bits=8,
            thermal_noise_sigma=5e-3,
            temperature=400.0,  # 高温
            leakage_rate=1e-3,
            memristor_drift=1e-5,
            transconductance_variation=0.1,
        )
    
    else:
        return AnalogCircuitConfig()




