"""
Dashboard快速启动脚本

自动设置Python路径并启动Streamlit Dashboard
"""
import os
import sys
from pathlib import Path

# 添加项目根目录到系统路径
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# 设置工作目录
os.chdir(PROJECT_ROOT)

if __name__ == "__main__":
    import subprocess
    
    dashboard_path = PROJECT_ROOT / "src" / "visualization" / "advanced_dashboard.py"
    
    print("=" * 60)
    print("启动模拟计算ANN训练可视化Dashboard")
    print("=" * 60)
    print(f"项目目录: {PROJECT_ROOT}")
    print(f"Dashboard: {dashboard_path}")
    print("\n正在启动Streamlit服务器...")
    print("=" * 60)
    
    # 启动Streamlit
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        str(dashboard_path),
        "--server.port=8501",
        "--server.headless=true"
    ])



