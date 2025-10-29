"""
包构建脚本

自动化包构建和检查流程
"""
import subprocess
import sys
import shutil
from pathlib import Path

def run_command(cmd, description):
    """运行命令并打印结果"""
    print(f"\n{'='*60}")
    print(f"{description}...")
    print(f"{'='*60}")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, shell=True)
        print(result.stdout)
        if result.stderr:
            print(result.stderr)
        print(f"[√] {description}成功")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[×] {description}失败")
        print(e.stdout)
        print(e.stderr)
        return False

def clean_build_dirs():
    """清理构建目录"""
    print("\n清理旧的构建文件...")
    dirs_to_clean = ['build', 'dist', '*.egg-info']
    
    for pattern in dirs_to_clean:
        for path in Path('.').glob(pattern):
            if path.is_dir():
                shutil.rmtree(path)
                print(f"  删除: {path}")
    
    print("[√] 清理完成")

def main():
    """主函数"""
    print("="*60)
    print("开始构建包")
    print("="*60)
    
    # 步骤1: 清理
    clean_build_dirs()
    
    # 步骤2: 安装构建工具
    if not run_command(
        "pip install --upgrade build twine",
        "安装构建工具"
    ):
        sys.exit(1)
    
    # 步骤3: 构建包
    if not run_command(
        "python -m build",
        "构建分发包"
    ):
        sys.exit(1)
    
    # 步骤4: 检查包
    if not run_command(
        "twine check dist/*",
        "检查包质量"
    ):
        sys.exit(1)
    
    # 完成
    print("\n" + "="*60)
    print("包构建完成！")
    print("="*60)
    print("\n生成的文件:")
    for file in Path('dist').glob('*'):
        print(f"  - {file}")
    
    print("\n下一步:")
    print("  1. 测试PyPI: twine upload --repository testpypi dist/*")
    print("  2. 正式PyPI: twine upload dist/*")
    print("  3. 或创建GitHub Release自动发布")

if __name__ == "__main__":
    main()

