# 快速发布指南

本文档提供最简洁的发布流程。完整指南请查看 [PUBLISHING_GUIDE.md](PUBLISHING_GUIDE.md)。

---

## 🚀 快速发布到PyPI

### 前置要求

1. 已注册 [PyPI](https://pypi.org) 账号
2. 已生成API Token（Account settings → API tokens）

### 发布步骤（5个命令）

```bash
# 1. 构建包
python scripts/build_package.py

# 2. 上传到Test PyPI（测试）
twine upload --repository testpypi dist/*
# 输入用户名: __token__
# 输入密码: [你的Test PyPI API Token]

# 3. 测试安装
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ analog-neural-training

# 4. 如果测试成功，发布到正式PyPI
twine upload dist/*
# 输入用户名: __token__
# 输入密码: [你的PyPI API Token]

# 5. 验证
pip install analog-neural-training
```

---

## 📦 快速发布到GitHub

### 步骤1: 创建仓库

```bash
# 初始化Git（如果还没有）
git init

# 添加所有文件
git add .

# 提交
git commit -m "Initial commit: v1.0.0"

# 创建GitHub仓库后，添加远程仓库
git remote add origin https://github.com/zhaoxuancan/analog-neural-training.git

# 推送
git branch -M main
git push -u origin main
```

### 步骤2: 配置GitHub Secrets

1. 进入仓库设置 → Secrets and variables → Actions
2. 添加两个Secrets:
   - `PYPI_API_TOKEN`: PyPI的API Token
   - `TEST_PYPI_API_TOKEN`: Test PyPI的API Token

### 步骤3: 创建Release（自动发布到PyPI）

```bash
# 创建tag
git tag v1.0.0
git push origin v1.0.0
```

然后在GitHub上创建Release，GitHub Actions会自动发布到PyPI。

---

## ✅ 发布前检查清单

```bash
# 1. 验证安装
python verify_installation.py

# 2. 检查版本号
grep "version" setup.py
grep "version" pyproject.toml

# 3. 检查CHANGELOG
cat CHANGELOG.md

# 4. 运行构建
python scripts/build_package.py
```

---

## 🆘 常见问题

### 包名冲突

修改`setup.py`和`pyproject.toml`中的`name`字段。

### 上传失败403

检查API Token是否正确：
- 用户名必须是: `__token__`
- 密码是你的API Token（以`pypi-`开头）

### 如何更新包

```bash
# 1. 更新版本号（setup.py, pyproject.toml）
# 2. 更新CHANGELOG.md
# 3. 重新构建和上传
python scripts/build_package.py
twine upload dist/*
```

---

## 📞 获取帮助

- 完整指南: [PUBLISHING_GUIDE.md](PUBLISHING_GUIDE.md)
- PyPI官方文档: https://packaging.python.org
- GitHub: https://github.com/zhaoxuancan/analog-neural-training/issues

---

**祝发布顺利！** 🎉

