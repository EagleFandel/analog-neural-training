"""
基准测试套件

全面对比传统优化器 vs 模拟计算启发式优化器
"""
from __future__ import annotations

import os
import sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from sklearn.datasets import fetch_openml, load_digits, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.models.mlp import MLP
from src.utils.seed import set_global_seed
from src.optim.analog_inspired import create_optimizer
from src.optim.baseline_adam import adam_train
from src.optim.baseline_gd import gd_train
from src.optim.baseline_rmsprop import rmsprop_train
from src.hardware.energy_models import compare_digital_vs_analog


class BenchmarkSuite:
    """基准测试套件"""
    
    def __init__(self, output_dir: str = "results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = []
    
    def run_mnist_benchmark(
        self,
        sample_size: int = 5000,
        steps: int = 100,
        seed: int = 0
    ):
        """MNIST 基准测试"""
        print("\n" + "="*60)
        print("MNIST 基准测试")
        print("="*60)
        
        set_global_seed(seed)
        
        # 加载数据
        try:
            data = fetch_openml("mnist_784", version=1, as_frame=False)
            x = data.data.astype(np.float64) / 255.0
            y = data.target.astype(int)
        except Exception:
            d = load_digits()
            x = d.data.astype(np.float64) / 16.0
            y = d.target.astype(int)
            sample_size = min(sample_size, x.shape[0])
        
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, train_size=min(sample_size, len(x)-100), stratify=y, random_state=seed
        )
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
        
        input_dim = x_train.shape[1]
        num_classes = int(np.max(y_train)) + 1
        model = MLP([input_dim, 128, 64, num_classes])
        
        # 测试所有优化器
        optimizers_to_test = [
            ("Adam", "baseline"),
            ("SGD", "baseline"),
            ("RMSProp", "baseline"),
            ("RK4", "analog"),
            ("DOPRI54", "analog"),
            ("IMEX", "analog"),
            ("Symplectic", "analog"),
            ("SDE", "analog"),
        ]
        
        for opt_name, opt_type in optimizers_to_test:
            print(f"\n测试 {opt_name}...")
            result = self._run_single_optimizer(
                opt_name, opt_type, model, x_train, y_train, x_test, y_test,
                steps=steps, lr=1e-3, task="classification"
            )
            result["dataset"] = "mnist"
            result["sample_size"] = len(x_train)
            self.results.append(result)
    
    def run_synthetic_benchmark(
        self,
        n_samples: int = 1000,
        n_features: int = 50,
        steps: int = 200,
        seed: int = 0
    ):
        """合成数据基准测试（分类）"""
        print("\n" + "="*60)
        print("合成数据基准测试")
        print("="*60)
        
        set_global_seed(seed)
        
        x, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=30,
            n_redundant=10,
            n_classes=3,
            random_state=seed
        )
        
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=seed
        )
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
        
        model = MLP([n_features, 32, 16, 3])
        
        optimizers = [("Adam", "baseline"), ("RK4", "analog"), ("Symplectic", "analog")]
        
        for opt_name, opt_type in optimizers:
            print(f"\n测试 {opt_name}...")
            result = self._run_single_optimizer(
                opt_name, opt_type, model, x_train, y_train, x_test, y_test,
                steps=steps, lr=1e-3, task="classification"
            )
            result["dataset"] = "synthetic"
            result["n_features"] = n_features
            self.results.append(result)
    
    def run_sine_regression_benchmark(
        self,
        n_samples: int = 200,
        steps: int = 300,
        seed: int = 0
    ):
        """正弦回归基准测试"""
        print("\n" + "="*60)
        print("正弦回归基准测试")
        print("="*60)
        
        set_global_seed(seed)
        
        x = np.linspace(-np.pi, np.pi, n_samples).reshape(-1, 1)
        y = np.sin(x).flatten() + 0.1 * np.random.randn(n_samples)
        
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=seed
        )
        
        model = MLP([1, 32, 32, 1])
        
        optimizers = [
            ("SGD", "baseline"),
            ("RK4", "analog"),
            ("DOPRI54", "analog"),
        ]
        
        for opt_name, opt_type in optimizers:
            print(f"\n测试 {opt_name}...")
            result = self._run_single_optimizer(
                opt_name, opt_type, model, x_train, y_train, x_test, y_test,
                steps=steps, lr=1e-2, task="regression"
            )
            result["dataset"] = "sine"
            self.results.append(result)
    
    def _run_single_optimizer(
        self,
        opt_name: str,
        opt_type: str,
        model: MLP,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
        steps: int,
        lr: float,
        task: str
    ) -> Dict:
        """运行单个优化器"""
        theta0 = model.theta0.copy()
        
        start_time = time.perf_counter()
        
        if opt_type == "baseline":
            if opt_name == "Adam":
                theta_final, losses = adam_train(
                    model.loss_and_grad, theta0, x_train, y_train, steps, lr, task
                )
                nfe = steps  # Adam每步1次梯度评估
            elif opt_name == "SGD":
                theta_final, losses = gd_train(
                    model.loss_and_grad, theta0, x_train, y_train, steps, lr, task
                )
                nfe = steps
            elif opt_name == "RMSProp":
                theta_final, losses = rmsprop_train(
                    model.loss_and_grad, theta0, x_train, y_train, steps, lr, task
                )
                nfe = steps
            else:
                raise ValueError(f"Unknown baseline: {opt_name}")
        
        else:  # analog
            optimizer = create_optimizer(
                opt_name.lower(), model.loss_and_grad, theta0, lr, track_energy=True
            )
            losses = []
            for _ in range(steps):
                _, loss = optimizer.step(x_train, y_train, task)
                losses.append(loss)
            theta_final = optimizer.theta
            nfe = optimizer.state.nfe_counter.count
        
        elapsed = time.perf_counter() - start_time
        
        # 评估
        final_loss, _ = model.loss_and_grad(theta_final, x_train, y_train, task)
        
        if task == "classification":
            preds_train = model.forward(theta_final, x_train, task)
            preds_test = model.forward(theta_final, x_test, task)
            train_acc = np.mean(np.argmax(preds_train, axis=1) == y_train)
            test_acc = np.mean(np.argmax(preds_test, axis=1) == y_test)
            metric = test_acc
        else:
            preds_test = model.forward(theta_final, x_test, task)
            mse = np.mean((preds_test.flatten() - y_test) ** 2)
            metric = -mse  # 负MSE，越大越好
            train_acc, test_acc = 0.0, 0.0
        
        print(f"  {opt_name}: 最终损失={final_loss:.4f}, NFE={nfe}, 时间={elapsed:.2f}s, 指标={metric:.4f}")
        
        return {
            "optimizer": opt_name,
            "type": opt_type,
            "final_loss": float(final_loss),
            "train_acc": float(train_acc),
            "test_acc": float(test_acc),
            "metric": float(metric),
            "nfe": int(nfe),
            "time_seconds": float(elapsed),
            "steps": steps,
            "loss_history": [float(l) for l in losses],
        }
    
    def save_results(self, filename: str = "benchmark_results.json"):
        """保存结果到JSON"""
        output_path = self.output_dir / filename
        with open(output_path, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"\n结果已保存到: {output_path}")
    
    def generate_summary_table(self) -> str:
        """生成汇总表格（LaTeX格式）"""
        if not self.results:
            return ""
        
        # 按数据集分组
        datasets = {}
        for r in self.results:
            ds = r.get("dataset", "unknown")
            if ds not in datasets:
                datasets[ds] = []
            datasets[ds].append(r)
        
        latex = "\\begin{table}[h]\n\\centering\n"
        latex += "\\begin{tabular}{lcccc}\n\\hline\n"
        latex += "优化器 & 最终损失 & NFE & 时间(s) & 准确率/指标 \\\\\n\\hline\n"
        
        for ds_name, results in datasets.items():
            latex += f"\\multicolumn{{5}}{{c}}{{\\textbf{{{ds_name}}}}} \\\\\n"
            for r in results:
                latex += f"{r['optimizer']} & {r['final_loss']:.4f} & {r['nfe']} & {r['time_seconds']:.2f} & {r['metric']:.4f} \\\\\n"
            latex += "\\hline\n"
        
        latex += "\\end{tabular}\n\\caption{基准测试结果}\n\\end{table}"
        
        return latex
    
    def save_latex_table(self, filename: str = "benchmark_table.tex"):
        """保存LaTeX表格"""
        output_path = self.output_dir / filename
        with open(output_path, "w") as f:
            f.write(self.generate_summary_table())
        print(f"LaTeX表格已保存到: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="运行完整基准测试套件")
    parser.add_argument("--mnist", action="store_true", help="运行MNIST测试")
    parser.add_argument("--synthetic", action="store_true", help="运行合成数据测试")
    parser.add_argument("--sine", action="store_true", help="运行正弦回归测试")
    parser.add_argument("--all", action="store_true", help="运行所有测试")
    parser.add_argument("--steps", type=int, default=100, help="训练步数")
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    parser.add_argument("--output", type=str, default="results", help="输出目录")
    
    args = parser.parse_args()
    
    suite = BenchmarkSuite(output_dir=args.output)
    
    if args.all or args.mnist:
        suite.run_mnist_benchmark(steps=args.steps, seed=args.seed)
    
    if args.all or args.synthetic:
        suite.run_synthetic_benchmark(steps=args.steps, seed=args.seed)
    
    if args.all or args.sine:
        suite.run_sine_regression_benchmark(steps=args.steps, seed=args.seed)
    
    # 保存结果
    suite.save_results()
    suite.save_latex_table()
    
    # 能耗对比
    print("\n" + "="*60)
    print("能耗对比分析")
    print("="*60)
    from src.hardware.energy_models import print_energy_comparison
    print_energy_comparison(num_params=50000, num_steps=args.steps, batch_size=32)


if __name__ == "__main__":
    main()




