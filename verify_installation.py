"""
安装验证脚本

快速验证项目是否正确安装并可以正常运行
"""
import sys
from pathlib import Path

# 添加项目根目录到Python路径
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

def test_imports():
    """测试核心模块是否可以导入"""
    print("1. 测试模块导入...")
    
    try:
        # 核心模块
        from src.models.mlp import MLP
        from src.optim.analog_inspired import RK4Optimizer
        
        # 分析模块
        from analysis.pl_condition import PLConditionVerifier
        from analysis.lyapunov_analysis import LyapunovAnalyzer
        from analysis.energy_drift import EnergyDriftAnalyzer
        
        # PDF导出
        from src.pdf_export.reportlab_renderer import ReportLabExporter
        
        print("   [√] 核心模块导入成功")
        return True
    except Exception as e:
        print(f"   [×] 模块导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_basic_training():
    """测试基本训练功能"""
    print("\n2. 测试基本训练...")
    
    try:
        import numpy as np
        from src.models.mlp import MLP
        from src.optim.analog_inspired import RK4Optimizer
        
        # 创建简单模型
        model = MLP([10, 5, 2])
        
        # 创建优化器
        optimizer = RK4Optimizer(
            model.loss_and_grad,
            model.theta0,
            lr=1e-3
        )
        
        # 准备测试数据
        x = np.random.randn(20, 10)
        y = np.random.randint(0, 2, 20)
        
        # 执行几步训练
        for _ in range(3):
            theta, loss = optimizer.step(x, y, task="classification")
        
        print(f"   [√] 训练成功（最终损失: {loss:.4f}）")
        return True
    except Exception as e:
        print(f"   [×] 训练失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_hardware_simulation():
    """测试硬件仿真"""
    print("\n3. 测试硬件仿真...")
    
    try:
        import numpy as np
        # 硬件仿真模块存在即可
        import src.hardware.analog_simulator
        import src.hardware.energy_models
        
        print("   [√] 硬件仿真模块可用")
        return True
    except Exception as e:
        print(f"   [×] 硬件仿真失败: {e}")
        return False


def test_theoretical_analysis():
    """测试理论分析工具"""
    print("\n4. 测试理论分析工具...")
    
    try:
        import numpy as np
        from analysis.pl_condition import PLConditionVerifier
        from analysis.lyapunov_analysis import LyapunovAnalyzer
        from analysis.energy_drift import EnergyDriftAnalyzer
        
        # 简单二次函数
        A = np.eye(3)
        
        def loss_fn(x):
            return 0.5 * np.dot(x, np.dot(A, x))
        
        def grad_fn(x):
            return np.dot(A, x)
        
        # PL条件验证
        verifier = PLConditionVerifier(loss_fn, grad_fn, optimal_value=0.0)
        sample_points = [np.random.randn(3) for _ in range(10)]
        pl_result = verifier.verify(sample_points)
        
        # Lyapunov分析
        trajectory = [np.random.randn(3) * np.exp(-0.1*i) for i in range(20)]
        lyap_analyzer = LyapunovAnalyzer(loss_fn)
        lyap_result = lyap_analyzer.analyze_trajectory(trajectory)
        
        # 能量漂移
        positions = [np.random.randn(3) for _ in range(20)]
        velocities = [np.random.randn(3) for _ in range(20)]
        
        def hamiltonian(x, v):
            return loss_fn(x) + 0.5 * np.sum(v**2)
        
        drift_analyzer = EnergyDriftAnalyzer(hamiltonian)
        drift_result = drift_analyzer.analyze_trajectory(positions, velocities)
        
        print("   [√] 理论分析工具成功")
        return True
    except Exception as e:
        print(f"   [×] 理论分析失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pdf_export():
    """测试PDF导出"""
    print("\n5. 测试PDF导出...")
    
    try:
        from src.pdf_export.models import ReportContext, Section, TableArtifact
        from src.pdf_export.reportlab_renderer import ReportLabExporter
        from pathlib import Path
        
        # 创建简单报告上下文
        section = Section(
            title="测试章节",
            subtitle="Test Section",
            summary="这是一个测试章节",
            figures=[],
            tables=[
                TableArtifact(
                    title="测试表格",
                    description="测试表格描述",
                    headers=["列1", "列2"],
                    rows=[["数据1", "数据2"]]
                )
            ]
        )
        
        context = ReportContext(
            title="验证测试报告",
            subtitle="Installation Verification Report",
            dataset_name="测试数据集",
            generated_at="2025-10-29",
            metadata={"作者": "自动测试", "版本": "1.0"},
            sections=[section]
        )
        
        # 生成PDF
        output_path = Path("verify_test_report.pdf")
        exporter = ReportLabExporter(output_path)
        exporter.generate(context)
        
        # 验证文件存在
        if output_path.exists():
            # 删除测试文件
            output_path.unlink()
            print("   [√] PDF导出成功")
            return True
        else:
            print("   [×] PDF文件未生成")
            return False
    except Exception as e:
        print(f"   [×] PDF导出失败: {e}")
        return False


def main():
    """主函数"""
    print("=" * 60)
    print("开始验证安装...")
    print("=" * 60)
    
    results = []
    
    # 运行所有测试
    results.append(("模块导入", test_imports()))
    results.append(("基本训练", test_basic_training()))
    results.append(("硬件仿真", test_hardware_simulation()))
    results.append(("理论分析", test_theoretical_analysis()))
    results.append(("PDF导出", test_pdf_export()))
    
    # 打印总结
    print("\n" + "=" * 60)
    print("验证结果总结:")
    print("=" * 60)
    
    total = len(results)
    passed = sum(1 for _, success in results if success)
    
    for name, success in results:
        status = "[√]" if success else "[×]"
        print(f"{status} {name}")
    
    print("\n" + "-" * 60)
    print(f"通过: {passed}/{total}")
    
    if passed == total:
        print("\n所有测试通过！安装成功！")
        return 0
    else:
        print(f"\n警告: {total - passed} 个测试失败")
        print("请检查依赖是否正确安装: pip install -r requirements.txt")
        return 1


if __name__ == "__main__":
    sys.exit(main())

