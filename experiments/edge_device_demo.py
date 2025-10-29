"""
边缘设备场景演示

模拟功耗受限、内存受限、实时约束的边缘设备训练场景
"""
from __future__ import annotations

import os
import sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List
import numpy as np
from sklearn.datasets import load_digits, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.models.mlp import MLP
from src.utils.seed import set_global_seed
from src.optim.analog_inspired import create_optimizer
from src.optim.baseline_adam import adam_train
from src.hardware.constrained_training import ConstrainedTrainer, HardwareConstraints
from src.hardware.analog_simulator import AnalogCircuitSimulator, create_realistic_config
from src.hardware.energy_models import HybridEnergyModel


class EdgeDeviceScenario:
    """边缘设备训练场景"""
    
    def __init__(
        self,
        scenario_name: str = "iot_sensor",
        output_dir: str = "results"
    ):
        self.scenario_name = scenario_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 场景配置
        self.scenarios = {
            "iot_sensor": {
                "description": "IoT传感器在线学习",
                "energy_budget_j": 10.0,  # 10 Joules
                "power_limit_w": 0.5,  # 500 mW
                "max_latency_ms": 100,  # 100ms
                "max_memory_mb": 1.0,  # 1 MB
                "circuit_config": "low_power",
            },
            "smartphone": {
                "description": "手机端模型微调",
                "energy_budget_j": 50.0,  # 50 Joules
                "power_limit_w": 2.0,  # 2 W
                "max_latency_ms": 50,
                "max_memory_mb": 10.0,  # 10 MB
                "circuit_config": "high_precision",
            },
            "wearable": {
                "description": "可穿戴设备持续学习",
                "energy_budget_j": 5.0,  # 5 Joules
                "power_limit_w": 0.2,  # 200 mW
                "max_latency_ms": 200,
                "max_memory_mb": 0.5,  # 0.5 MB
                "circuit_config": "low_power",
            },
            "drone": {
                "description": "无人机强化学习",
                "energy_budget_j": 100.0,  # 100 Joules
                "power_limit_w": 5.0,  # 5 W
                "max_latency_ms": 20,  # 实时控制
                "max_memory_mb": 5.0,
                "circuit_config": "harsh_environment",
            },
        }
    
    def run_scenario(self, scenario_name: str, seed: int = 0):
        """运行特定场景"""
        if scenario_name not in self.scenarios:
            raise ValueError(f"未知场景: {scenario_name}")
        
        config = self.scenarios[scenario_name]
        print("\n" + "="*60)
        print(f"边缘设备场景: {config['description']}")
        print("="*60)
        print(f"能耗预算: {config['energy_budget_j']} J")
        print(f"功耗限制: {config['power_limit_w']*1000} mW")
        print(f"延迟约束: {config['max_latency_ms']} ms/步")
        print(f"内存限制: {config['max_memory_mb']} MB")
        print("-"*60)
        
        set_global_seed(seed)
        
        # 准备数据（小规模）
        x_train, x_test, y_train, y_test = self._prepare_data(seed)
        
        # 创建小模型
        input_dim = x_train.shape[1]
        num_classes = len(np.unique(y_train))
        
        # 根据内存限制选择模型大小
        max_params = int(config['max_memory_mb'] * 1024 * 1024 / 8)  # float64 = 8 bytes
        hidden_size = min(64, max_params // (input_dim + num_classes))
        
        model = MLP([input_dim, hidden_size, num_classes])
        print(f"模型大小: {model.theta0.size} 参数 ({model.theta0.nbytes/1024/1024:.2f} MB)")
        
        # 硬件约束
        constraints = HardwareConstraints(
            energy_budget_joules=config['energy_budget_j'],
            power_limit_watts=config['power_limit_w'],
            max_latency_per_step_ms=config['max_latency_ms'],
            max_param_memory_mb=config['max_memory_mb']
        )
        
        # 模拟电路
        circuit_config = create_realistic_config(config['circuit_config'])
        simulator = AnalogCircuitSimulator(circuit_config, seed=seed)
        
        # 能耗模型
        energy_model = HybridEnergyModel(analog_compute_ratio=0.8)
        
        # 测试不同优化器
        optimizers_to_test = [
            ("Adam", "baseline"),
            ("RK4", "analog"),
            ("DOPRI54", "analog"),
            ("Symplectic", "analog"),
        ]
        
        results = []
        
        for opt_name, opt_type in optimizers_to_test:
            print(f"\n测试 {opt_name} 优化器...")
            result = self._run_constrained_training(
                opt_name, opt_type, model, x_train, y_train, x_test, y_test,
                constraints, simulator, energy_model
            )
            results.append(result)
        
        # 保存结果
        self._save_results(scenario_name, results)
        
        # 打印对比
        self._print_comparison(results)
    
    def _prepare_data(self, seed: int):
        """准备小规模数据集"""
        # 使用sklearn digits（小规模）
        data = load_digits()
        x = data.data.astype(np.float64) / 16.0
        y = data.target.astype(int)
        
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=seed
        )
        
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
        
        return x_train, x_test, y_train, y_test
    
    def _run_constrained_training(
        self,
        opt_name: str,
        opt_type: str,
        model: MLP,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
        constraints: HardwareConstraints,
        simulator: AnalogCircuitSimulator,
        energy_model: HybridEnergyModel
    ) -> Dict:
        """在约束下运行训练"""
        theta0 = model.theta0.copy()
        
        if opt_type == "baseline":
            # 基线优化器（不使用硬件仿真）
            start_time = time.perf_counter()
            
            # 手动实现能耗预算控制
            losses = []
            theta = theta0.copy()
            energy_consumed = 0.0
            step = 0
            
            while energy_consumed < constraints.energy_budget_joules:
                # Adam更新
                loss, grad = model.loss_and_grad(theta, x_train, y_train, "classification")
                losses.append(loss)
                
                # 简单的Adam步骤（简化版）
                theta = theta - 1e-3 * grad
                
                # 估算能耗
                step_energy = energy_model.compute_training_step_energy(
                    theta.size, x_train.shape[0], "adam"
                )["total"]
                energy_consumed += step_energy
                step += 1
                
                if step > 1000:  # 防止无限循环
                    break
            
            elapsed = time.perf_counter() - start_time
            theta_final = theta
            nfe = step
            
        else:  # analog
            # 根据优化器类型调整学习率
            if opt_name.lower() == "rk4":
                lr = 1e-4  # RK4需要更小的学习率
            elif opt_name.lower() == "dopri54":
                lr = 5e-4  # DOPRI54中等学习率
            elif opt_name.lower() == "symplectic":
                lr = 5e-4  # Symplectic需要适中学习率
            else:
                lr = 1e-3
            
            optimizer = create_optimizer(
                opt_name.lower(),
                model.loss_and_grad,
                theta0,
                lr=lr,
                track_energy=True
            )
            
            trainer = ConstrainedTrainer(
                optimizer, constraints, simulator, energy_model, verbose=False
            )
            
            losses = []
            step = 0
            start_time = time.perf_counter()
            
            while trainer.can_continue() and step < 1000:
                theta, loss, stats = trainer.step(x_train, y_train, "classification")
                losses.append(loss)
                step += 1
                
                if step % 50 == 0:
                    remaining = stats.get("budget_remaining", {})
                    if "energy_percent" in remaining:
                        print(f"  步骤 {step}: 损失={loss:.4f}, 剩余能耗={remaining['energy_percent']:.1f}%")
            
            elapsed = time.perf_counter() - start_time
            theta_final = optimizer.theta
            nfe = optimizer.state.nfe_counter.count
            energy_consumed = trainer.total_energy_consumed
            summary = trainer.get_summary()
        
        # 评估
        preds_test = model.forward(theta_final, x_test, "classification")
        test_acc = np.mean(np.argmax(preds_test, axis=1) == y_test)
        
        final_loss = losses[-1] if losses else float('inf')
        
        print(f"  完成: 步数={step}, NFE={nfe}, 能耗={energy_consumed:.3f}J, 准确率={test_acc:.4f}")
        
        return {
            "optimizer": opt_name,
            "type": opt_type,
            "steps_completed": step,
            "nfe": nfe,
            "energy_consumed_j": float(energy_consumed),
            "time_seconds": float(elapsed),
            "final_loss": float(final_loss),
            "test_accuracy": float(test_acc),
            "loss_history": [float(l) for l in losses],
            "efficiency": float(test_acc / energy_consumed) if energy_consumed > 0 else 0,  # 准确率/能耗
        }
    
    def _save_results(self, scenario_name: str, results: List[Dict]):
        """保存结果"""
        output_file = self.output_dir / f"edge_device_{scenario_name}.json"
        with open(output_file, "w") as f:
            json.dump({
                "scenario": scenario_name,
                "scenario_config": self.scenarios[scenario_name],
                "results": results
            }, f, indent=2)
        print(f"\n结果已保存到: {output_file}")
    
    def _print_comparison(self, results: List[Dict]):
        """打印对比"""
        print("\n" + "="*60)
        print("性能对比")
        print("="*60)
        print(f"{'优化器':<15} {'步数':>8} {'能耗(J)':>10} {'准确率':>10} {'效率':>10}")
        print("-"*60)
        
        for r in results:
            print(f"{r['optimizer']:<15} {r['steps_completed']:>8} "
                  f"{r['energy_consumed_j']:>10.3f} {r['test_accuracy']:>10.4f} "
                  f"{r['efficiency']:>10.4f}")
        
        print("="*60)
        
        # 找到最佳效率
        best = max(results, key=lambda r: r['efficiency'])
        print(f"\n最佳效率: {best['optimizer']} (准确率/能耗 = {best['efficiency']:.4f})")


def main():
    parser = argparse.ArgumentParser(description="边缘设备训练场景演示")
    parser.add_argument(
        "--scenario",
        type=str,
        default="iot_sensor",
        choices=["iot_sensor", "smartphone", "wearable", "drone", "all"],
        help="选择场景"
    )
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    parser.add_argument("--output", type=str, default="results", help="输出目录")
    
    args = parser.parse_args()
    
    demo = EdgeDeviceScenario(output_dir=args.output)
    
    if args.scenario == "all":
        for scenario_name in demo.scenarios.keys():
            demo.run_scenario(scenario_name, seed=args.seed)
    else:
        demo.run_scenario(args.scenario, seed=args.seed)


if __name__ == "__main__":
    main()

