"""
PDF导出依赖安装辅助脚本（ReportLab方案）

自动检测并安装PDF导出所需的Python包
"""
import os
import platform
import subprocess
import sys


def check_pip_package(package_name):
    """检查Python包是否已安装"""
    try:
        __import__(package_name)
        return True
    except ImportError:
        return False


def install_pip_packages():
    """安装Python依赖包"""
    packages = ["reportlab>=4.0", "pyyaml>=6.0"]
    
    print("=" * 60)
    print("安装PDF导出依赖（ReportLab方案）...")
    print("=" * 60)
    
    for pkg in packages:
        pkg_name = pkg.split(">=")[0]
        if check_pip_package(pkg_name):
            print(f"[OK] {pkg_name} 已安装")
        else:
            print(f"[安装中] 正在安装 {pkg}...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
                print(f"[OK] {pkg_name} 安装成功")
            except subprocess.CalledProcessError:
                print(f"[错误] {pkg_name} 安装失败")
                return False
    
    return True


def check_system_dependencies():
    """检查系统依赖并提供安装指导"""
    system = platform.system()
    
    print("\n" + "=" * 60)
    print(f"检测到操作系统: {system}")
    print("=" * 60)
    
    print("\n[信息] ReportLab是纯Python库，无需额外系统依赖。")
    print("\n所有操作系统（Windows / Linux / macOS）均可直接使用。")
    
    # 可选：中文字体支持提示
    print("\n[可选] 中文字体支持：")
    print("当前版本使用Helvetica字体（不支持中文）。")
    print("如需中文支持，可参考ReportLab文档配置自定义字体。")


def test_installation():
    """测试安装是否成功"""
    print("\n" + "=" * 60)
    print("测试安装...")
    print("=" * 60)
    
    # 测试ReportLab
    print("\n测试ReportLab（PDF生成）...")
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import A4
        import io
        
        # 创建简单的PDF测试
        pdf_buffer = io.BytesIO()
        c = canvas.Canvas(pdf_buffer, pagesize=A4)
        c.drawString(100, 750, "ReportLab Test")
        c.save()
        
        if len(pdf_buffer.getvalue()) > 0:
            print("[OK] ReportLab 工作正常")
        else:
            print("[错误] ReportLab PDF生成失败")
            return False
    except Exception as e:
        print(f"[错误] ReportLab 测试失败: {e}")
        return False
    
    # 测试PyYAML
    print("\n测试PyYAML（配置文件解析）...")
    try:
        import yaml
        test_yaml = "key: value"
        data = yaml.safe_load(test_yaml)
        if data.get("key") == "value":
            print("[OK] PyYAML 工作正常")
        else:
            print("[错误] PyYAML 解析失败")
            return False
    except Exception as e:
        print(f"[错误] PyYAML 测试失败: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("[成功] 所有测试通过！PDF导出功能已就绪")
    print("=" * 60)
    
    return True


def main():
    """主函数"""
    print("PDF导出功能依赖安装程序")
    print("=" * 60)
    
    # 1. 安装Python包
    if not install_pip_packages():
        print("\n❌ Python包安装失败，请手动安装或检查网络连接")
        return False
    
    # 2. 检查并提示系统依赖
    check_system_dependencies()
    
    # 3. 测试安装
    print("\n按回车键继续测试安装...")
    input()
    
    success = test_installation()
    
    if success:
        print("\n[成功] 安装完成！现在可以在Dashboard中使用PDF导出功能了")
        print("\n快速开始：")
        print("  # Dashboard导出")
        print("  streamlit run src/visualization/advanced_dashboard.py")
        print("\n  # 命令行演示")
        print("  python src/pdf_export/cli_demo.py")
    else:
        print("\n[警告] 安装未完全成功，请根据上述错误信息进行排查")
        print("\n故障排除：")
        print("  1. 确保Python版本 >= 3.8")
        print("  2. 尝试升级pip: python -m pip install --upgrade pip")
        print("  3. 手动安装: pip install reportlab pyyaml")
        print("\n详细文档: docs/pdf_export_guide.md")
    
    return success


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n[取消] 安装已取消")
        sys.exit(1)
    except Exception as e:
        print(f"\n[错误] 发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

