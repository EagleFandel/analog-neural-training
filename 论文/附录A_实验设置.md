# 附录A 实验设置与硬件/软件环境

## A1 软件环境
- Python: 3.8+
- 主要库：
  - numpy: 1.24+
  - matplotlib: 3.7+
  - scipy: 1.11+
  - scikit-learn: 1.3+
  - pandas: 2.0+
  - plotly: 5.15+
  - streamlit: 1.32+
  - reportlab: 4.0+
（完整依赖见项目根目录`requirements.txt`）

## A2 硬件环境（用于时间与能耗基准估算）
- CPU: Intel Core i7-10700 (或同级)
- GPU: NVIDIA GeForce RTX 3070 (或同级)
- RAM: 32 GB DDR4
- OS: Windows 10/11 or Ubuntu 20.04+

## A3 超参数设置（工程案例）
- 学习率搜索网格：{1e-3, 5e-4, 1e-4}
- 批量大小：128（MNIST子集）/ 64（正弦回归）
- 训练步数：100–300（依任务）
- 随机种子：42
- DOPRI54容差tol：{1e-3, 5e-4, 1e-4}
- IMEX隐式子步：2
- 辛积分γ（动量）：0.1
（详见`experiments/`目录下的各实验脚本）

## A4 能耗模型参数（示意）
- 数字能耗：基于典型45nm工艺节点的FLOPs与内存访问能耗估算；
- 模拟能耗：基于跨导放大器与电容的典型值，ADC/DAC能耗与位宽成指数关系。
（详见`src/hardware/energy_models.py`）


