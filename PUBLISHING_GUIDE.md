# PyPI发布和GitHub仓库设置指南

本指南详细说明如何将项目发布到PyPI并设置GitHub仓库。

---

## 📋 目录

1. [准备工作](#准备工作)
2. [GitHub仓库设置](#github仓库设置)
3. [PyPI发布流程](#pypi发布流程)
4. [版本管理](#版本管理)
5. [常见问题](#常见问题)

---

## 准备工作

### 1. 检查项目完整性

运行验证脚本确保项目正常工作：

```bash
python verify_installation.py
```

应该看到所有5项测试通过。

### 2. 更新版本号

在以下文件中更新版本号：

- `setup.py` (line 17): `version="1.0.0"`
- `pyproject.toml` (line 7): `version = "1.0.0"`
- `src/__init__.py`: 添加 `__version__ = "1.0.0"`

### 3. 更新CHANGELOG.md

记录本次发布的所有更改。

### 4. 准备PyPI账号

1. 注册PyPI账号: https://pypi.org/account/register/
2. 注册Test PyPI账号: https://test.pypi.org/account/register/
3. 配置API Token（在下面的步骤中说明）

---

## GitHub仓库设置

### 步骤1: 创建GitHub仓库

1. 访问 https://github.com/new
2. 仓库名称: `analog-neural-training`
3. 描述: `模拟计算启发式神经网络训练系统：基于ODE积分器的高能效训练框架`
4. 选择 Public（公开）
5. **不要** 初始化README、.gitignore或LICENSE（我们已经有了）
6. 点击 "Create repository"

### 步骤2: 本地Git初始化

```bash
# 初始化Git仓库（如果还没有）
git init

# 添加所有文件
git add .

# 创建初始提交
git commit -m "Initial commit: v1.0.0"

# 添加远程仓库
git remote add origin https://github.com/zhaoxuancan/analog-neural-training.git

# 推送到GitHub
git branch -M main
git push -u origin main
```

### 步骤3: 配置GitHub Secrets

为了使GitHub Actions能够发布到PyPI，需要设置Secrets：

1. 访问仓库设置: `https://github.com/zhaoxuancan/analog-neural-training/settings/secrets/actions`

2. 点击 "New repository secret"

3. 添加以下Secrets：

   **a. PyPI API Token**
   - Name: `PYPI_API_TOKEN`
   - Value: [从PyPI获取的API Token]
   
   **如何获取PyPI API Token:**
   - 登录 https://pypi.org/
   - 进入 Account settings → API tokens
   - 点击 "Add API token"
   - Token name: `analog-neural-training`
   - Scope: "Entire account" 或 "Project: analog-neural-training"
   - 复制生成的token（只显示一次！）

   **b. Test PyPI API Token**
   - Name: `TEST_PYPI_API_TOKEN`
   - Value: [从Test PyPI获取的API Token]
   
   **如何获取Test PyPI API Token:**
   - 登录 https://test.pypi.org/
   - 同样的步骤获取API token

### 步骤4: 设置GitHub Pages（可选）

如果要发布文档：

1. 进入仓库设置 → Pages
2. Source: 选择 `main` 分支的 `/docs` 文件夹
3. 点击 Save

---

## PyPI发布流程

### 方法1: 手动发布（推荐首次发布）

#### 步骤1: 安装构建工具

```bash
pip install --upgrade build twine
```

#### 步骤2: 清理旧的构建文件

```bash
# Windows
rmdir /s /q build dist *.egg-info

# Linux/Mac
rm -rf build/ dist/ *.egg-info/
```

#### 步骤3: 构建分发包

```bash
python -m build
```

这会创建两个文件：
- `dist/analog-neural-training-1.0.0.tar.gz` (源代码分发)
- `dist/analog_neural_training-1.0.0-py3-none-any.whl` (wheel分发)

#### 步骤4: 检查包

```bash
twine check dist/*
```

应该显示：`Checking dist/analog-neural-training-1.0.0.tar.gz: PASSED` 等

#### 步骤5: 上传到Test PyPI（测试）

```bash
twine upload --repository testpypi dist/*
```

输入你的Test PyPI用户名和密码（或使用`__token__`作为用户名和API token作为密码）。

#### 步骤6: 从Test PyPI测试安装

```bash
# 创建新的虚拟环境测试
python -m venv test_env
source test_env/bin/activate  # Windows: test_env\Scripts\activate

# 从Test PyPI安装
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ analog-neural-training

# 测试导入
python -c "from src.models.mlp import MLP; print('Success!')"

# 退出测试环境
deactivate
```

#### 步骤7: 发布到正式PyPI

如果Test PyPI测试成功：

```bash
twine upload dist/*
```

输入你的PyPI用户名和密码（或使用API token）。

#### 步骤8: 验证发布

访问 https://pypi.org/project/analog-neural-training/

安装测试：

```bash
pip install analog-neural-training
```

### 方法2: 使用GitHub Actions自动发布

#### 步骤1: 创建Git Tag

```bash
git tag v1.0.0
git push origin v1.0.0
```

#### 步骤2: 创建GitHub Release

1. 访问 `https://github.com/zhaoxuancan/analog-neural-training/releases/new`
2. 选择刚才创建的tag: `v1.0.0`
3. Release title: `v1.0.0 - 初始发布`
4. 描述框中填写release notes（从CHANGELOG.md复制）:

```markdown
## 新功能

- 5种模拟启发优化器（RK4, DOPRI54, IMEX, Symplectic, SDE）
- 硬件仿真器
- 能耗分析工具
- 理论分析工具（PL条件、Lyapunov、能量漂移）
- 可视化Dashboard
- PDF报告生成
- 4个应用案例

## 完整更新日志

查看 [CHANGELOG.md](CHANGELOG.md) 了解详细信息。
```

5. 勾选 "Set as the latest release"
6. 点击 "Publish release"

GitHub Actions会自动触发，构建并发布到PyPI。

#### 步骤3: 监控发布

1. 进入 Actions 标签页
2. 查看 "Publish to PyPI" workflow的运行状态
3. 如果成功，几分钟后就能在PyPI上看到你的包

---

## 版本管理

### 语义化版本控制

遵循 [SemVer](https://semver.org/) 规范：

- `MAJOR.MINOR.PATCH` (例如 `1.0.0`)
- **MAJOR**: 不兼容的API更改
- **MINOR**: 向后兼容的新功能
- **PATCH**: 向后兼容的bug修复

### 版本更新流程

1. **更新版本号**:
   ```bash
   # 在setup.py和pyproject.toml中更新
   # 例如从1.0.0更新到1.1.0
   ```

2. **更新CHANGELOG.md**:
   ```markdown
   ## [1.1.0] - 2025-11-01
   
   ### 新增
   - 新功能描述
   
   ### 修复
   - Bug修复描述
   ```

3. **提交更改**:
   ```bash
   git add .
   git commit -m "Bump version to 1.1.0"
   git push
   ```

4. **创建新tag**:
   ```bash
   git tag v1.1.0
   git push origin v1.1.0
   ```

5. **创建GitHub Release**（会自动触发发布）

---

## 常见问题

### Q1: `twine upload` 失败，提示403错误

**A**: 检查API token是否正确，确保：
- 使用`__token__`作为用户名
- API token作为密码
- Token有正确的权限

### Q2: 包名已被占用

**A**: 修改`setup.py`和`pyproject.toml`中的`name`字段，例如：
- `analog-neural-training` → `analog-nn-training-zxc`

### Q3: GitHub Actions失败

**A**: 检查：
1. Secrets是否正确设置
2. 查看Actions日志了解具体错误
3. 确保`requirements.txt`中的所有依赖都可用

### Q4: 如何撤回已发布的版本

**A**: PyPI不允许删除已发布的版本，但可以：
1. 标记为"yanked"（不推荐安装但仍可用）
2. 发布新的补丁版本

### Q5: 如何更新PyPI上的项目描述

**A**: 修改`README.md`后重新发布新版本即可自动更新。

### Q6: 如何添加徽章到README

在README.md顶部添加：

```markdown
[![PyPI version](https://badge.fury.io/py/analog-neural-training.svg)](https://badge.fury.io/py/analog-neural-training)
[![Downloads](https://pepy.tech/badge/analog-neural-training)](https://pepy.tech/project/analog-neural-training)
[![CI](https://github.com/zhaoxuancan/analog-neural-training/workflows/CI/badge.svg)](https://github.com/zhaoxuancan/analog-neural-training/actions)
```

---

## 发布检查清单

在发布前确保：

- [ ] 所有测试通过 (`python verify_installation.py`)
- [ ] 版本号已更新（setup.py, pyproject.toml）
- [ ] CHANGELOG.md已更新
- [ ] README.md是最新的
- [ ] 所有代码已提交到Git
- [ ] GitHub仓库已创建并推送
- [ ] GitHub Secrets已配置
- [ ] 在Test PyPI测试成功
- [ ] 创建Git tag
- [ ] 创建GitHub Release
- [ ] PyPI发布成功
- [ ] 从PyPI安装测试成功

---

## 后续维护

### 1. 监控Issues和Pull Requests

定期检查GitHub Issues和PR，及时响应用户反馈。

### 2. 更新依赖

定期更新`requirements.txt`中的依赖版本：

```bash
pip list --outdated
```

### 3. 安全更新

使用GitHub Dependabot自动检测安全漏洞。

### 4. 社区建设

- 回复Issues
- 审查Pull Requests
- 更新文档
- 发布Release Notes

---

## 资源链接

- **PyPI**: https://pypi.org/project/analog-neural-training/
- **Test PyPI**: https://test.pypi.org/project/analog-neural-training/
- **GitHub**: https://github.com/zhaoxuancan/analog-neural-training
- **PyPI打包指南**: https://packaging.python.org/tutorials/packaging-projects/
- **GitHub Actions文档**: https://docs.github.com/en/actions

---

**祝发布顺利！** 🚀

如有问题，请参考官方文档或在Issues中提问。

