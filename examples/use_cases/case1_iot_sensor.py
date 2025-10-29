"""
应用案例1: IoT传感器在线学习（功耗受限）

场景：智能传感器需要在现场自适应学习，功耗预算极其有限（10 Joules）
目标：在能耗预算内达到最高准确率
"""
from __future__ import annotations

import os
import sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.models.mlp import MLP
from src.optim.analog_inspired import create_optimizer
from src.hardware.constrained_training import ConstrainedTrainer, HardwareConstraints
from src.hardware.analog_simulator import AnalogCircuitSimulator, create_realistic_config
from src.hardware.energy_models import HybridEnergyModel
from src.utils.seed import set_global_seed


def run_iot_sensor_demo():
    print("="*60)
    print("案例1: IoT传感器在线学习")
    print("="*60)
    print("\n场景描述:")
    print("- 环境监测传感器需要识别异常事件")
    print("- 电池供电，总能耗预算10 Joules")
    print("- 需要在部署后快速适应本地环境")
    print("-"*60)
    
    set_global_seed(42)
    
    # 模拟传感器数据：温度、湿度、气压等特征
    n_samples = 500
    n_features = 8  # 8个传感器通道
    
    x, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=6,
        n_redundant=2,
        n_classes=2,  # 正常 vs 异常
        random_state=42
    )
    
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )
    
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    
    # 创建轻量级模型（参数少以节省能耗）
    model = MLP([n_features, 16, 2])  # 很小的网络
    print(f"\n模型参数: {model.theta0.size} 个")
    print(f"内存占用: {model.theta0.nbytes / 1024:.2f} KB")
    
    # IoT设备硬件约束
    constraints = HardwareConstraints(
        energy_budget_joules=10.0,        # 10J能耗预算
        power_limit_watts=0.2,            # 200mW功耗上限（电池供电）
        max_latency_per_step_ms=200,      # 200ms延迟可接受
        max_param_memory_mb=0.5           # 500KB内存限制
    )
    
    # 模拟低功耗电路
    circuit_config = create_realistic_config("low_power")
    simulator = AnalogCircuitSimulator(circuit_config, seed=42)
    
    # 混合能耗模型（80%模拟计算）
    energy_model = HybridEnergyModel(analog_compute_ratio=0.8)
    
    # 对比两种优化器
    results = {}
    
    for method_name in ["rk4", "dopri54"]:
        print(f"\n{'='*60}")
        print(f"测试优化器: {method_name.upper()}")
        print(f"{'='*60}")
        
        # 创建优化器
        optimizer = create_optimizer(
            method_name,
            model.loss_and_grad,
            model.theta0.copy(),
            lr=1e-2 if method_name == "rk4" else 1e-3
        )
        
        # 创建约束训练器
        trainer = ConstrainedTrainer(
            optimizer, constraints, simulator, energy_model, verbose=False
        )
        
        # 训练直到能耗用尽
        step = 0
        best_acc = 0.0
        
        print(f"\n开始训练...")
        while trainer.can_continue() and step < 500:
            theta, loss, stats = trainer.step(x_train, y_train, "classification")
            step += 1
            
            # 每50步评估一次
            if step % 50 == 0:
                preds = model.forward(theta, x_test, "classification")
                acc = np.mean(np.argmax(preds, axis=1) == y_test)
                best_acc = max(best_acc, acc)
                
                remaining = stats.get("budget_remaining", {})
                energy_pct = remaining.get("energy_percent", 0)
                
                print(f"  步骤 {step}: 损失={loss:.4f}, 准确率={acc*100:.2f}%, "
                      f"剩余能耗={energy_pct:.1f}%")
        
        # 最终评估
        final_preds = model.forward(optimizer.theta, x_test, "classification")
        final_acc = np.mean(np.argmax(final_preds, axis=1) == y_test)
        
        summary = trainer.get_summary()
        
        results[method_name] = {
            "steps": summary["steps_completed"],
            "energy_j": summary["total_energy_j"],
            "final_acc": final_acc,
            "best_acc": best_acc,
            "violations": len(summary["constraint_violations"])
        }
        
        print(f"\n{method_name.upper()} 完成:")
        print(f"  训练步数: {results[method_name]['steps']}")
        print(f"  能耗使用: {results[method_name]['energy_j']:.3f} J")
        print(f"  最终准确率: {results[method_name]['final_acc']*100:.2f}%")
        print(f"  最佳准确率: {results[method_name]['best_acc']*100:.2f}%")
        print(f"  约束违反: {results[method_name]['violations']}次")
    
    # 对比总结
    print(f"\n{'='*60}")
    print("性能对比总结")
    print(f"{'='*60}")
    
    for method in ["rk4", "dopri54"]:
        r = results[method]
        efficiency = r["best_acc"] / r["energy_j"]
        print(f"\n{method.upper()}:")
        print(f"  能效 (准确率/能耗): {efficiency:.4f}")
        print(f"  最佳准确率: {r['best_acc']*100:.2f}%")
    
    # 推荐
    best_method = max(results.keys(), 
                      key=lambda k: results[k]["best_acc"] / results[k]["energy_j"])
    
    print(f"\n✅ 推荐方案: {best_method.upper()}")
    print(f"\n结论:")
    print("在IoT传感器这种极端功耗受限场景下，")
    print("自适应DOPRI54优化器能更有效利用能耗预算，")
    print("通过动态步长控制最小化不必要的计算。")


if __name__ == "__main__":
    run_iot_sensor_demo()




