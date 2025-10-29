# 部署指令 - 推送到GitHub和发布到PyPI

本文档提供具体的部署命令，基于您的GitHub仓库：
**https://github.com/EagleFandel/analog-neural-training.git**

---

## 📦 步骤1: 推送代码到GitHub

### 1.1 初始化Git仓库（如果还没有）

```bash
cd "D:\Documents\Projects\ANN AI Training"

# 初始化Git
git init

# 配置Git用户信息（如果还没配置）
git config user.name "EagleFandel"
git config user.email "zhaoxuancan@example.com"
```

### 1.2 添加所有文件

```bash
# 添加所有文件
git add .

# 查看将要提交的文件
git status
```

### 1.3 创建初始提交

```bash
git commit -m "Initial commit: v1.0.0 - Complete analog neural training system

- 5 ODE-based optimizers (RK4, DOPRI54, IMEX, Symplectic, SDE)
- Hardware simulator and energy models
- Theoretical analysis tools (PL condition, Lyapunov, energy drift)
- Visualization dashboard with PDF export
- 4 application case studies
- Complete documentation and examples"
```

### 1.4 关联远程仓库

```bash
# 添加远程仓库
git remote add origin https://github.com/EagleFandel/analog-neural-training.git

# 验证远程仓库
git remote -v
```

### 1.5 推送到GitHub

```bash
# 设置主分支为main
git branch -M main

# 推送到GitHub
git push -u origin main
```

如果遇到认证问题，GitHub现在需要使用Personal Access Token：

1. 访问：https://github.com/settings/tokens
2. 点击 "Generate new token (classic)"
3. 勾选 `repo` 权限
4. 生成token并保存（只显示一次！）
5. 使用token作为密码推送

---

## 🔑 步骤2: 配置GitHub Secrets（用于自动发布）

### 2.1 获取PyPI API Token

1. **注册PyPI账号**（如果还没有）
   - 访问：https://pypi.org/account/register/
   - 填写信息并验证邮箱

2. **生成API Token**
   - 登录后访问：https://pypi.org/manage/account/token/
   - 点击 "Add API token"
   - Token name: `analog-neural-training`
   - Scope: "Entire account"（首次发布）或选择特定项目
   - 点击 "Add token"
   - **立即复制token**（格式：`pypi-xxxxx`，只显示一次！）

3. **同样操作获取Test PyPI Token**
   - 访问：https://test.pypi.org/account/register/
   - 访问：https://test.pypi.org/manage/account/token/
   - 生成并保存token

### 2.2 在GitHub仓库中添加Secrets

1. 访问：https://github.com/EagleFandel/analog-neural-training/settings/secrets/actions

2. 点击 "New repository secret"

3. 添加第一个Secret：
   - Name: `PYPI_API_TOKEN`
   - Secret: [粘贴PyPI的API token]
   - 点击 "Add secret"

4. 添加第二个Secret：
   - Name: `TEST_PYPI_API_TOKEN`
   - Secret: [粘贴Test PyPI的API token]
   - 点击 "Add secret"

---

## 🚀 步骤3: 发布到PyPI

### 方法A: 通过GitHub Release自动发布（推荐）

```bash
# 1. 创建版本tag
git tag -a v1.0.0 -m "Release v1.0.0: Initial release"

# 2. 推送tag到GitHub
git push origin v1.0.0
```

然后：

1. 访问：https://github.com/EagleFandel/analog-neural-training/releases/new
2. 选择tag: `v1.0.0`
3. Release title: `v1.0.0 - 初始发布`
4. 描述框填写（从CHANGELOG.md复制）：

```markdown
## 🎉 首次发布

模拟计算启发式神经网络训练系统 v1.0.0

### ✨ 主要特性

#### 五种ODE优化器
- **RK4** - 高精度四阶龙格-库塔
- **DOPRI54** - 自适应步长优化器
- **IMEX** - 半隐式方法（处理刚性问题）
- **Symplectic** - 辛积分保能量
- **SDE** - 随机微分方程（噪声鲁棒）

#### 完整工具链
- 硬件仿真器（ADC/DAC量化、热噪声、电容泄漏）
- 能耗分析（数字vs模拟架构对比）
- 理论分析工具（PL条件验证、Lyapunov稳定性、能量漂移）
- 可视化Dashboard（Streamlit + Plotly）
- PDF报告生成（ReportLab）

#### 丰富示例
- 基准测试套件
- 边缘设备场景演示
- 4个实际应用案例
- 理论分析演示

### 📦 安装

```bash
pip install analog-neural-training
```

### 📚 快速开始

查看 [GETTING_STARTED.md](GETTING_STARTED.md)

### 📖 文档

- [用户指南](docs/user_guide.md)
- [理论背景](docs/theory.md)
- [API文档](docs/user_guide.md)

### 🙏 致谢

感谢所有贡献者和使用者！

---

**完整更新日志**: [CHANGELOG.md](CHANGELOG.md)
```

5. 勾选 "Set as the latest release"
6. 点击 "Publish release"

**GitHub Actions会自动触发，构建并发布到PyPI！**

### 方法B: 手动发布到PyPI

```bash
# 1. 安装构建工具
pip install --upgrade build twine

# 2. 构建包
python scripts/build_package.py

# 3. 先上传到Test PyPI测试
twine upload --repository testpypi dist/*
# 用户名: __token__
# 密码: [你的Test PyPI token]

# 4. 测试安装
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ analog-neural-training

# 5. 测试成功后，上传到正式PyPI
twine upload dist/*
# 用户名: __token__
# 密码: [你的PyPI token]
```

---

## ✅ 验证发布

### 验证GitHub

访问：https://github.com/EagleFandel/analog-neural-training

应该看到所有文件已推送。

### 验证PyPI

1. 访问：https://pypi.org/project/analog-neural-training/
2. 安装测试：

```bash
# 创建新环境测试
python -m venv test_env
source test_env/bin/activate  # Windows: test_env\Scripts\activate

# 从PyPI安装
pip install analog-neural-training

# 运行验证
python -c "from src.models.mlp import MLP; print('Success!')"

# 退出测试环境
deactivate
```

---

## 📋 发布后的检查清单

- [ ] GitHub仓库代码已推送
- [ ] GitHub Actions CI通过
- [ ] PyPI包已发布
- [ ] 从PyPI安装测试成功
- [ ] README.md在GitHub显示正常
- [ ] Release notes已发布
- [ ] 徽章显示正常

---

## 🎯 后续更新流程

当需要发布新版本时：

```bash
# 1. 更新版本号（setup.py, pyproject.toml, src/__init__.py）
# 2. 更新CHANGELOG.md
# 3. 提交更改
git add .
git commit -m "Bump version to 1.1.0"
git push

# 4. 创建新tag
git tag -a v1.1.0 -m "Release v1.1.0"
git push origin v1.1.0

# 5. 创建GitHub Release（会自动发布到PyPI）
```

---

## 🆘 常见问题

### Q: git push被拒绝

**A**: 可能需要使用Personal Access Token：
```bash
git remote set-url origin https://[YOUR_TOKEN]@github.com/EagleFandel/analog-neural-training.git
```

### Q: GitHub Actions失败

**A**: 检查：
1. Secrets是否正确配置
2. 查看Actions日志
3. 确保requirements.txt中所有依赖都存在

### Q: PyPI包名冲突

**A**: 如果`analog-neural-training`已被占用，修改为：
- `analog-nn-training-eaglefandel`
- `neural-analog-training`
- 等其他名称

---

## 📞 获取帮助

- GitHub Issues: https://github.com/EagleFandel/analog-neural-training/issues
- 完整指南: [PUBLISHING_GUIDE.md](PUBLISHING_GUIDE.md)

---

**准备开始部署！** 🚀

