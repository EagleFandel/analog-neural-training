"""
Setup script for analog-training package
"""
from setuptools import setup, find_packages
from pathlib import Path

# 读取README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# 读取requirements
requirements = (this_directory / "requirements.txt").read_text(encoding='utf-8').splitlines()
requirements = [r.strip() for r in requirements if r.strip() and not r.startswith('#')]

setup(
    name="analog-neural-training",
    version="1.0.0",
    author="Zhao Xuancan",
    author_email="zhaoxuancan@example.com",
    description="模拟计算启发式神经网络训练系统：基于ODE积分器的高能效训练框架",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/EagleFandel/analog-neural-training",
    project_urls={
        "Bug Tracker": "https://github.com/EagleFandel/analog-neural-training/issues",
        "Documentation": "https://github.com/EagleFandel/analog-neural-training/blob/main/docs/user_guide.md",
        "Source Code": "https://github.com/EagleFandel/analog-neural-training",
        "Changelog": "https://github.com/EagleFandel/analog-neural-training/blob/main/CHANGELOG.md",
    },
    packages=find_packages(include=['src', 'src.*', 'analysis', 'analysis.*']),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
        "Natural Language :: Chinese (Simplified)",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "pytorch": ["torch>=2.0.0"],
        "tensorflow": ["tensorflow>=2.10.0"],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=0.990",
        ],
        "viz": [
            "streamlit>=1.32.0",
            "plotly>=5.15.0",
        ],
    },
    # entry_points={
    #     "console_scripts": [
    #         "analog-verify=verify_installation:main",
    #     ],
    # },
    include_package_data=True,
    keywords=[
        "deep-learning",
        "neural-networks",
        "analog-computing",
        "ode-solvers",
        "energy-efficient-ai",
        "optimization",
        "neuromorphic-computing",
    ],
    license="MIT",
    zip_safe=False,
)





