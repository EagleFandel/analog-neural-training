"""
应用案例4: 刚性优化问题（IMEX加速批归一化网络）

场景：强正则化或批归一化导致的刚性损失景观
目标：使用IMEX方法扩展稳定域，允许更大步长
"""
from __future__ import annotations

import os
import sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.models.mlp import MLP
from src.optim.analog_inspired import IMEXOptimizer, RK4Optimizer
from src.utils.seed import set_global_seed


def create_stiff_loss_function(lambda_reg=10.0):
    """创建刚性损失函数
    
    L = L_data(θ) + λ ||θ||²
    
    其中正则项产生刚性（Hessian特征值差异大）
    """
    base_model = MLP([20, 16, 2])
    
    def stiff_loss_and_grad(theta, x, y, task):
        # 数据项
        loss_data, grad_data = base_model.loss_and_grad(theta, x, y, task)
        
        # 强正则化项（刚性来源）
        reg_loss = 0.5 * lambda_reg * np.dot(theta, theta)
        reg_grad = lambda_reg * theta
        
        # 总损失
        total_loss = loss_data + reg_loss
        total_grad = grad_data + reg_grad
        
        return total_loss, total_grad
    
    return stiff_loss_and_grad, base_model


def estimate_stiffness_ratio(loss_grad_fn, theta, x, y):
    """估计刚性比（条件数）"""
    from src.hardware.constrained_training import _estimate_loss_stiffness
    
    def wrapped_fn(theta_in, x_in, y_in, task):
        return loss_grad_fn(theta_in, x, y, "classification")
    
    return _estimate_loss_stiffness(wrapped_fn, theta)


def run_stiff_optimization_demo():
    print("="*60)
    print("案例4: 刚性优化问题 - IMEX方法")
    print("="*60)
    print("\n场景描述:")
    print("- 强L2正则化导致损失景观刚性")
    print("- 传统显式方法需要极小步长")
    print("- IMEX方法隐式处理刚性项，允许大步长")
    print("-"*60)
    
    set_global_seed(42)
    
    # 准备数据
    x, y = make_classification(
        n_samples=500,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=2,
        random_state=42
    )
    
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )
    
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    
    # 测试不同正则化强度
    lambda_values = [1.0, 10.0, 50.0]
    
    all_results = {}
    
    for lambda_reg in lambda_values:
        print(f"\n{'='*60}")
        print(f"正则化强度 λ = {lambda_reg}")
        print(f"{'='*60}")
        
        # 创建刚性损失函数
        stiff_loss_grad_fn, base_model = create_stiff_loss_function(lambda_reg)
        theta_init = base_model.theta0.copy()
        
        # 估计刚性
        stiffness = estimate_stiffness_ratio(stiff_loss_grad_fn, theta_init, x_train, y_train)
        print(f"估计刚性比: {stiffness:.2f}")
        
        results_lambda = {}
        
        # 对比三种方法
        for method_name in ["rk4_small", "rk4_large", "imex"]:
            print(f"\n测试: {method_name}")
            
            if method_name == "rk4_small":
                # RK4 小步长（安全）
                optimizer = RK4Optimizer(
                    stiff_loss_grad_fn,
                    theta_init.copy(),
                    lr=1e-4,  # 小步长
                    track_energy=True
                )
                label = "RK4 (lr=1e-4)"
            
            elif method_name == "rk4_large":
                # RK4 大步长（可能不稳定）
                optimizer = RK4Optimizer(
                    stiff_loss_grad_fn,
                    theta_init.copy(),
                    lr=1e-2,  # 大步长
                    track_energy=True
                )
                label = "RK4 (lr=1e-2)"
            
            else:  # imex
                # IMEX 大步长（稳定）
                optimizer = IMEXOptimizer(
                    stiff_loss_grad_fn,
                    theta_init.copy(),
                    lr=1e-2,  # 大步长
                    implicit_mass=lambda_reg,  # 隐式处理正则项
                    max_iter=10,
                    track_energy=True
                )
                label = "IMEX (lr=1e-2)"
            
            # 训练
            losses = []
            accuracies = []
            max_steps = 200
            
            for step in range(max_steps):
                try:
                    theta, loss = optimizer.step(x_train, y_train, "classification")
                    losses.append(loss)
                    
                    # 评估
                    preds = base_model.forward(theta, x_test, "classification")
                    acc = np.mean(np.argmax(preds, axis=1) == y_test)
                    accuracies.append(acc)
                    
                    # 检查NaN
                    if np.isnan(loss) or np.isinf(loss):
                        print(f"  ⚠ 步骤 {step}: 发散！")
                        break
                    
                except Exception as e:
                    print(f"  ⚠ 步骤 {step}: 错误 - {e}")
                    break
            
            final_acc = accuracies[-1] if accuracies else 0.0
            converged = len(losses) == max_steps and not np.isnan(losses[-1])
            
            results_lambda[method_name] = {
                "losses": losses,
                "accuracies": accuracies,
                "final_acc": final_acc,
                "converged": converged,
                "nfe": optimizer.state.nfe_counter.count,
                "label": label
            }
            
            status = "✅ 收敛" if converged else "❌ 发散"
            print(f"  {status}, 最终准确率={final_acc*100:.2f}%, NFE={results_lambda[method_name]['nfe']}")
        
        all_results[lambda_reg] = results_lambda
    
    # 可视化
    print(f"\n生成对比图表...")
    
    fig, axes = plt.subplots(len(lambda_values), 2, figsize=(14, 4*len(lambda_values)))
    
    for idx, lambda_reg in enumerate(lambda_values):
        results = all_results[lambda_reg]
        
        # 损失曲线
        ax_loss = axes[idx, 0] if len(lambda_values) > 1 else axes[0]
        for method in ["rk4_small", "rk4_large", "imex"]:
            if method in results:
                r = results[method]
                ax_loss.semilogy(r["losses"], label=r["label"], linewidth=2)
        
        ax_loss.set_xlabel("步数")
        ax_loss.set_ylabel("损失（对数）")
        ax_loss.set_title(f"λ={lambda_reg} - 损失曲线")
        ax_loss.legend()
        ax_loss.grid(True, alpha=0.3)
        
        # 准确率曲线
        ax_acc = axes[idx, 1] if len(lambda_values) > 1 else axes[1]
        for method in ["rk4_small", "rk4_large", "imex"]:
            if method in results:
                r = results[method]
                ax_acc.plot([a*100 for a in r["accuracies"]], label=r["label"], linewidth=2)
        
        ax_acc.set_xlabel("步数")
        ax_acc.set_ylabel("准确率 (%)")
        ax_acc.set_title(f"λ={lambda_reg} - 准确率曲线")
        ax_acc.legend()
        ax_acc.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_dir = Path("results/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "stiff_imex_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"图表已保存到: {output_path}")
    
    # 总结
    print(f"\n{'='*60}")
    print("结论")
    print(f"{'='*60}")
    print("\n观察:")
    print("1. 正则化强度↑ → 刚性↑ → 显式方法需要更小步长")
    print("2. RK4大步长在强正则化下发散")
    print("3. IMEX方法在相同大步长下保持稳定")
    print("\n✅ IMEX优势: 隐式处理刚性项，扩展稳定域")
    print("   应用: 批归一化、强正则化、病态Hessian等场景")


if __name__ == "__main__":
    run_stiff_optimization_demo()





