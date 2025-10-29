"""
理论分析工具包

提供优化算法的理论分析工具：
1. PL条件验证 - 验证Polyak-Łojasiewicz条件
2. Lyapunov分析 - 分析优化轨迹的稳定性
3. 能量漂移分析 - 分析辛积分器的能量守恒性
"""
from analysis.pl_condition import (
    PLConditionVerifier,
    PLVerificationResult,
    verify_quadratic_pl
)
from analysis.lyapunov_analysis import (
    LyapunovAnalyzer,
    EnergyLyapunovAnalyzer,
    LyapunovAnalysisResult,
    compare_optimizers_stability
)
from analysis.energy_drift import (
    EnergyDriftAnalyzer,
    LossKineticEnergyAnalyzer,
    EnergyDriftResult,
    compare_energy_conservation,
    analyze_symplectic_vs_nonsymplectic
)

__all__ = [
    # PL条件
    "PLConditionVerifier",
    "PLVerificationResult",
    "verify_quadratic_pl",
    # Lyapunov分析
    "LyapunovAnalyzer",
    "EnergyLyapunovAnalyzer",
    "LyapunovAnalysisResult",
    "compare_optimizers_stability",
    # 能量漂移
    "EnergyDriftAnalyzer",
    "LossKineticEnergyAnalyzer",
    "EnergyDriftResult",
    "compare_energy_conservation",
    "analyze_symplectic_vs_nonsymplectic",
]
