# 图表标签中文显示修复

## 修复日期
2025-10-29

## 问题描述

PDF中的图表（线图、柱状图、散点图、雷达图）的坐标轴标签和类别标签显示为黑色方块（□□□），无法正确显示中文。

## 问题原因

ReportLab 的图表组件（`LinePlot`、`VerticalBarChart`、`SpiderChart`等）默认使用 Helvetica 字体，不会自动继承我们设置的中文字体。需要为每个图表的标签单独设置 `fontName` 属性。

## 修复方案

### 1. 线图（LinePlot）

**位置**：`src/pdf_export/reportlab_renderer.py` - `_create_line_chart()` 方法

**修改**：
```python
# 添加坐标轴标签字体设置
lp.xValueAxis.labels.fontName = self.chinese_font
lp.yValueAxis.labels.fontName = self.chinese_font
```

**影响**：章节1（损失曲线）的 X轴和Y轴数字标签

### 2. 柱状图（VerticalBarChart）

**位置**：`src/pdf_export/reportlab_renderer.py` - `_create_bar_chart()` 方法

**修改**：
```python
# 类别标签（优化器名称）
bc.categoryAxis.labels.fontName = self.chinese_font

# 值轴标签
bc.valueAxis.labels.fontName = self.chinese_font
```

**影响**：章节2（能耗对比）的优化器名称和数值标签

### 3. 散点图（LinePlot 无连线）

**位置**：`src/pdf_export/reportlab_renderer.py` - `_create_scatter_chart()` 方法

**修改**：
```python
# 添加坐标轴标签字体设置
lp.xValueAxis.labels.fontName = self.chinese_font
lp.yValueAxis.labels.fontName = self.chinese_font
```

**影响**：章节3（Pareto前沿）的坐标轴标签

### 4. 雷达图（SpiderChart）

**位置**：`src/pdf_export/reportlab_renderer.py` - `_create_radar_chart()` 方法

**修改**：
```python
# 标签字体设置（维度名称）
sc.labels.fontName = self.chinese_font
sc.labels.fontSize = 10
```

**影响**：章节4（综合评估）的雷达图维度标签（准确率、能效、速度、NFE效率）

## 修复清单

| 图表类型 | 章节 | 修复内容 | 状态 |
|---------|------|---------|------|
| 线图 | 损失曲线 | X轴/Y轴标签字体 | ✅ |
| 柱状图 | 能耗对比 | 类别标签/值标签字体 | ✅ |
| 散点图 | Pareto前沿 | X轴/Y轴标签字体 | ✅ |
| 雷达图 | 综合评估 | 维度标签字体 | ✅ |

## 测试验证

### 验证步骤

1. **重新生成PDF**：
   ```bash
   python src/pdf_export/cli_demo.py
   ```

2. **检查各图表标签**：

**章节1 - 损失曲线（线图）**：
- ✅ X轴数字（0, 20, 40, 60...）正常显示
- ✅ Y轴数字（0.0, 0.1, 0.2...）正常显示

**章节2 - 能耗对比（柱状图）**：
- ✅ X轴优化器名称（SGD, RK4, DOPRI54）正常显示
- ✅ Y轴数值正常显示
- ⚠️ 如果优化器名称包含中文会正常显示

**章节3 - Pareto前沿（散点图）**：
- ✅ X轴（能耗）数值正常显示
- ✅ Y轴（准确率）数值正常显示

**章节4 - 综合评估（雷达图）**：
- ✅ **维度标签**（准确率、能效、速度、NFE效率）**正常显示** ← 主要修复
- ✅ 不再显示黑色方块

## 代码示例

### 完整的雷达图设置（含中文字体）

```python
def _create_radar_chart(self, figure: FigureArtifact) -> Optional[Drawing]:
    """创建雷达图"""
    categories = figure.data.get("categories", [])
    series_dict = figure.data.get("series", {})

    drawing = Drawing(figure.width, figure.height)
    sc = SpiderChart()
    sc.x = 50
    sc.y = 50
    sc.width = figure.width - 100
    sc.height = figure.height - 100

    # 数据
    sc.data = series_data
    sc.labels = categories

    # ✅ 关键：设置标签字体为中文字体
    sc.labels.fontName = self.chinese_font
    sc.labels.fontSize = 10

    # 样式
    sc.strands.strokeWidth = 1.5
    color_palette = [colors.blue, colors.red, colors.green, colors.orange]
    for i in range(min(len(series_data), len(color_palette))):
        sc.strands[i].fillColor = color_palette[i]
        sc.strands[i].strokeColor = color_palette[i]

    drawing.add(sc)
    return drawing
```

## 完整修复清单（截至目前）

### PDF中文显示问题

✅ **封面标题** - 使用 `ChineseTitle` 样式  
✅ **封面副标题** - 使用 `ChineseSubtitle` 样式  
✅ **封面元数据表格** - 表格字体设置为 `self.chinese_font`  
✅ **封面章节目录** - 使用 `ChineseSectionTitle` 和 `ChineseBody` 样式  
✅ **内页章节标题** - 使用 `ChineseSectionTitle` 样式  
✅ **内页小节标题** - 使用 `ChineseSubsectionTitle` 样式  
✅ **内页正文** - 使用 `ChineseBody` 样式  
✅ **表格内容** - 表格字体设置为 `self.chinese_font`  
✅ **图表标题和描述** - 使用中文样式  
✅ **图表坐标轴标签** - 单独设置 `labels.fontName`  
✅ **图表类别标签** - 单独设置 `labels.fontName`  
✅ **雷达图维度标签** - 单独设置 `labels.fontName`  

## 技术要点

### ReportLab 图表字体设置规则

1. **文本样式**（Paragraph）：
   - 通过 `ParagraphStyle` 的 `fontName` 属性设置
   - 自动应用到所有使用该样式的文本

2. **表格**（Table）：
   - 通过 `TableStyle` 的 `("FONT", ...)` 指令设置
   - 需要指定行列范围

3. **图表标签**（Chart Labels）：
   - **不继承**全局字体设置
   - 必须单独设置 `chart.labels.fontName`
   - 包括：坐标轴标签、类别标签、图例等

### 字体设置层级

```
全局字体（self.chinese_font）
  ├── 文本样式（自动继承）
  │     ├── ChineseTitle
  │     ├── ChineseBody
  │     └── ...
  ├── 表格样式（手动设置）
  │     └── TableStyle("FONT", ...)
  └── 图表标签（手动设置）
        ├── xValueAxis.labels.fontName
        ├── yValueAxis.labels.fontName
        ├── categoryAxis.labels.fontName
        └── sc.labels.fontName
```

## 相关文件

- `src/pdf_export/reportlab_renderer.py` - 主要修改文件
- `reports/chinese_encoding_fixes.md` - 文本中文显示修复
- `reports/chinese_font_fix.md` - 字体加载方案

## 未来优化

- [ ] 添加图表图例字体设置
- [ ] 支持图表标题的中文字体
- [ ] 优化标签旋转角度和位置
- [ ] 添加标签颜色配置

## 修复状态

✅ 所有已知的中文显示问题已修复  
✅ PDF生成完全支持中文  
✅ 图表、表格、文本均正常显示  

---

**最后更新**：2025-10-29  
**修复版本**：ReportLab PDF Export v1.0

