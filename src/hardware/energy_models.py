"""
能耗模型库

详细的数字和模拟计算能耗模型，用于对比分析
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np


@dataclass
class DigitalEnergyModel:
    """数字计算能耗模型
    
    基于FLOPs和内存访问的能耗估算
    """
    # 能耗常数（基于45nm CMOS工艺）
    energy_per_flop: float = 1.0e-12  # 1 pJ per FLOP
    energy_per_mac: float = 3.7e-12  # 3.7 pJ per MAC (乘加)
    energy_per_sram_read: float = 5.0e-12  # 5 pJ per SRAM read
    energy_per_sram_write: float = 10.0e-12  # 10 pJ per SRAM write
    energy_per_dram_read: float = 640.0e-12  # 640 pJ per DRAM read
    energy_per_dram_write: float = 800.0e-12  # 800 pJ per DRAM write
    
    # 动态功耗
    clock_frequency: float = 1e9  # 1 GHz
    dynamic_power: float = 1.0  # W (待机功耗)
    
    def compute_flops_energy(self, num_flops: int) -> float:
        """计算浮点运算能耗"""
        return float(num_flops) * self.energy_per_flop
    
    def compute_mac_energy(self, num_macs: int) -> float:
        """计算乘加运算能耗（更精确）"""
        return float(num_macs) * self.energy_per_mac
    
    def compute_memory_energy(
        self,
        sram_reads: int = 0,
        sram_writes: int = 0,
        dram_reads: int = 0,
        dram_writes: int = 0
    ) -> float:
        """计算内存访问能耗"""
        energy = 0.0
        energy += sram_reads * self.energy_per_sram_read
        energy += sram_writes * self.energy_per_sram_write
        energy += dram_reads * self.energy_per_dram_read
        energy += dram_writes * self.energy_per_dram_write
        return energy
    
    def compute_forward_backward_energy(
        self,
        num_params: int,
        batch_size: int,
        use_cache: bool = True
    ) -> float:
        """计算前向+反向传播的能耗
        
        Args:
            num_params: 参数数量
            batch_size: 批大小
            use_cache: 是否使用缓存（SRAM vs DRAM）
        """
        # 前向：2 * num_params 次MAC（权重乘法+偏置加法）
        forward_macs = 2 * num_params * batch_size
        
        # 反向：约3倍于前向（梯度计算）
        backward_macs = 3 * forward_macs
        
        compute_energy = self.compute_mac_energy(forward_macs + backward_macs)
        
        # 内存访问
        if use_cache:
            # 参数从SRAM读取
            mem_energy = self.compute_memory_energy(sram_reads=num_params * 4)
        else:
            # 参数从DRAM读取
            mem_energy = self.compute_memory_energy(dram_reads=num_params * 4)
        
        return compute_energy + mem_energy
    
    def compute_optimizer_step_energy(
        self,
        num_params: int,
        optimizer_type: str = "adam"
    ) -> float:
        """计算优化器更新步的能耗
        
        Args:
            optimizer_type: "sgd" | "adam" | "rmsprop"
        """
        if optimizer_type == "sgd":
            # θ = θ - lr * grad (2次操作)
            flops = 2 * num_params
        elif optimizer_type == "adam":
            # 动量、二阶矩估计、偏差修正（约10次操作）
            flops = 10 * num_params
        elif optimizer_type == "rmsprop":
            # 滑动平均（约5次操作）
            flops = 5 * num_params
        else:
            flops = 2 * num_params
        
        compute_energy = self.compute_flops_energy(flops)
        
        # 内存访问（读梯度、读参数、写参数）
        mem_energy = self.compute_memory_energy(
            sram_reads=2 * num_params,
            sram_writes=num_params
        )
        
        return compute_energy + mem_energy


@dataclass
class AnalogEnergyModel:
    """模拟计算能耗模型
    
    基于模拟电路元件的能耗估算
    """
    # 跨导放大器（Transconductance Amplifier）
    gm_power: float = 10e-6  # 10 μW per gm cell
    
    # 运算放大器
    opamp_power: float = 100e-6  # 100 μW per opamp
    
    # 电容充放电
    capacitor_voltage: float = 1.0  # V
    capacitance: float = 1e-12  # 1 pF
    
    # ADC/DAC 转换
    adc_energy_per_sample: float = 100e-12  # 100 pJ per sample (8-bit)
    dac_energy_per_sample: float = 50e-12  # 50 pJ per sample
    
    # 忆阻器
    memristor_read_energy: float = 1e-15  # 1 fJ per read
    memristor_write_energy: float = 1e-12  # 1 pJ per write
    
    def compute_analog_mac_energy(self, num_macs: int) -> float:
        """模拟乘加能耗（跨导阵列）
        
        在模拟域，MAC通过KCL自然完成，几乎无能耗
        主要能耗来自跨导放大器的静态功耗
        """
        # 假设每个MAC对应一个跨导单元，工作1μs
        time_per_mac = 1e-6  # 1 μs
        energy = num_macs * self.gm_power * time_per_mac
        return energy
    
    def compute_capacitor_energy(self, num_updates: int) -> float:
        """电容充放电能耗
        
        E = 1/2 * C * V^2
        """
        energy_per_update = 0.5 * self.capacitance * (self.capacitor_voltage ** 2)
        return num_updates * energy_per_update
    
    def compute_adc_dac_energy(
        self,
        num_adc_samples: int,
        num_dac_samples: int,
        bits: int = 8
    ) -> float:
        """ADC/DAC 转换能耗
        
        能耗与位宽指数相关：E ∝ 2^n
        """
        # 位宽调整
        bit_factor = 2 ** (bits - 8)
        
        adc_energy = num_adc_samples * self.adc_energy_per_sample * bit_factor
        dac_energy = num_dac_samples * self.dac_energy_per_sample * bit_factor
        
        return adc_energy + dac_energy
    
    def compute_memristor_energy(
        self,
        num_reads: int,
        num_writes: int
    ) -> float:
        """忆阻器读写能耗"""
        read_energy = num_reads * self.memristor_read_energy
        write_energy = num_writes * self.memristor_write_energy
        return read_energy + write_energy
    
    def compute_forward_backward_energy(
        self,
        num_params: int,
        batch_size: int,
        adc_bits: int = 8
    ) -> float:
        """模拟前向+反向传播能耗"""
        # MAC在模拟域几乎免费
        macs = 2 * num_params * batch_size
        mac_energy = self.compute_analog_mac_energy(macs)
        
        # ADC: 输出需要转换为数字
        # DAC: 输入需要转换为模拟
        conversion_energy = self.compute_adc_dac_energy(
            num_adc_samples=num_params,  # 输出层
            num_dac_samples=num_params * batch_size,  # 输入
            bits=adc_bits
        )
        
        return mac_energy + conversion_energy
    
    def compute_optimizer_step_energy(
        self,
        num_params: int,
        use_memristor: bool = True
    ) -> float:
        """模拟优化器更新步能耗
        
        在模拟域，参数更新通过电容充放电完成
        """
        if use_memristor:
            # 忆阻器：读当前值，写新值
            return self.compute_memristor_energy(
                num_reads=num_params,
                num_writes=num_params
            )
        else:
            # 电容：充放电
            return self.compute_capacitor_energy(num_params)


@dataclass
class HybridEnergyModel:
    """混合数字-模拟架构能耗模型"""
    
    digital_model: DigitalEnergyModel
    analog_model: AnalogEnergyModel
    
    # 架构参数
    analog_compute_ratio: float = 0.8  # 80%的计算在模拟域
    
    def __init__(
        self,
        digital_model: Optional[DigitalEnergyModel] = None,
        analog_model: Optional[AnalogEnergyModel] = None,
        analog_compute_ratio: float = 0.8
    ):
        self.digital_model = digital_model or DigitalEnergyModel()
        self.analog_model = analog_model or AnalogEnergyModel()
        self.analog_compute_ratio = analog_compute_ratio
    
    def compute_training_step_energy(
        self,
        num_params: int,
        batch_size: int,
        optimizer_type: str = "adam",
        adc_bits: int = 8
    ) -> Dict[str, float]:
        """计算一个训练步的总能耗，分解为各部分"""
        
        # 前向+反向（混合）
        analog_macs = int(2 * num_params * batch_size * self.analog_compute_ratio)
        digital_macs = 2 * num_params * batch_size - analog_macs
        
        forward_backward_analog = self.analog_model.compute_forward_backward_energy(
            num_params, batch_size, adc_bits
        ) * self.analog_compute_ratio
        
        forward_backward_digital = self.digital_model.compute_mac_energy(digital_macs)
        
        # 优化器更新（数字端）
        optimizer_energy = self.digital_model.compute_optimizer_step_energy(
            num_params, optimizer_type
        )
        
        total = forward_backward_analog + forward_backward_digital + optimizer_energy
        
        return {
            "forward_backward_analog": forward_backward_analog,
            "forward_backward_digital": forward_backward_digital,
            "optimizer": optimizer_energy,
            "total": total,
            "speedup_vs_digital": (
                self.digital_model.compute_forward_backward_energy(num_params, batch_size) +
                self.digital_model.compute_optimizer_step_energy(num_params, optimizer_type)
            ) / total if total > 0 else 1.0,
        }


def compare_digital_vs_analog(
    num_params: int,
    num_steps: int,
    batch_size: int = 32
) -> Dict[str, float]:
    """对比纯数字 vs 纯模拟 vs 混合架构的能耗
    
    Returns:
        包含各架构总能耗和加速比的字典
    """
    digital_model = DigitalEnergyModel()
    analog_model = AnalogEnergyModel()
    hybrid_model = HybridEnergyModel()
    
    # 纯数字
    digital_energy_per_step = (
        digital_model.compute_forward_backward_energy(num_params, batch_size) +
        digital_model.compute_optimizer_step_energy(num_params, "adam")
    )
    digital_total = digital_energy_per_step * num_steps
    
    # 纯模拟
    analog_energy_per_step = (
        analog_model.compute_forward_backward_energy(num_params, batch_size) +
        analog_model.compute_optimizer_step_energy(num_params, use_memristor=True)
    )
    analog_total = analog_energy_per_step * num_steps
    
    # 混合
    hybrid_stats = hybrid_model.compute_training_step_energy(num_params, batch_size)
    hybrid_total = hybrid_stats["total"] * num_steps
    
    return {
        "digital_total_joules": digital_total,
        "analog_total_joules": analog_total,
        "hybrid_total_joules": hybrid_total,
        "analog_speedup": digital_total / analog_total if analog_total > 0 else float('inf'),
        "hybrid_speedup": digital_total / hybrid_total if hybrid_total > 0 else float('inf'),
        "energy_savings_analog": (1 - analog_total / digital_total) * 100 if digital_total > 0 else 0,
        "energy_savings_hybrid": (1 - hybrid_total / digital_total) * 100 if digital_total > 0 else 0,
    }


def print_energy_comparison(
    num_params: int,
    num_steps: int,
    batch_size: int = 32
):
    """打印能耗对比报告"""
    results = compare_digital_vs_analog(num_params, num_steps, batch_size)
    
    print("=" * 60)
    print(f"能耗对比报告")
    print(f"参数量: {num_params:,}, 训练步数: {num_steps:,}, 批大小: {batch_size}")
    print("=" * 60)
    print(f"纯数字架构:  {results['digital_total_joules']:.6f} J")
    print(f"纯模拟架构:  {results['analog_total_joules']:.6f} J  (加速 {results['analog_speedup']:.1f}×)")
    print(f"混合架构:    {results['hybrid_total_joules']:.6f} J  (加速 {results['hybrid_speedup']:.1f}×)")
    print("-" * 60)
    print(f"模拟架构节能: {results['energy_savings_analog']:.1f}%")
    print(f"混合架构节能: {results['energy_savings_hybrid']:.1f}%")
    print("=" * 60)




