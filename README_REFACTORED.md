# PDFTranslator 重构架构说明

## 整体架构

重构后的代码按照你描述的整体逻辑组织，包含三个核心组件：

### 1. VLAdapter (视觉大模型适配器)

**文件**: `vladapter.py`

包含两个核心模块：

#### a. 输入转换器 (策略模式)
- `InputConverterStrategy`: 抽象基类
- `PictToPictConverter`: 图片到图片转换器（默认策略，不做任何转换）
- `PdfToPictConverter`: PDF到图片转换器

#### b. 视觉推理模型
- `VisionModel`: 真正的视觉推理模型
- 接口: `translate(pict_path) -> (List[str], List[str], int, int)`
  - 第一个返回值: 包含HTML标签的内容
  - 第二个返回值: 去除HTML标签的内容
  - 第三个返回值: 输入高度
  - 第四个返回值: 输入宽度

### 2. LL (语言大模型)

**文件**: `translator.py` (保持不变)

负责翻译，你已经写好了，不需要修改。

### 3. Persistence (持久化层)

**文件**: `persistence.py`

支持两种策略：

#### a. LaTeX输出策略
- `LatexPersistenceStrategy`: 输出LaTeX文件（你现有的实现）

#### b. PDF输出策略
- `PdfPersistenceStrategy`: 输出PDF文件（空壳实现，你自己写）

## 主要文件说明

### 核心文件

1. **`vladapter.py`**: VLAdapter实现
   - `VLAdapter`: 主适配器类
   - `InputConverterStrategy`: 输入转换器策略基类
   - `VisionModel`: 视觉推理模型

2. **`persistence.py`**: 持久化层实现
   - `PersistenceStrategy`: 持久化策略基类
   - `LatexPersistenceStrategy`: LaTeX输出策略
   - `PdfPersistenceStrategy`: PDF输出策略（空壳）
   - `ImageProcessor`: 图片处理器

3. **`main_refactored.py`**: 重构后的主程序
   - `DocumentProcessor`: 文档处理器，整合所有组件
   - `main()`: 主函数

4. **`example_refactored.py`**: 使用示例

### 辅助文件

- `translator.py`: 语言大模型（保持不变）
- `pdf_to_image.py`: PDF转图片工具（保持不变）

## 使用方法

### 1. 命令行使用

```bash
# 处理图片
python main_refactored.py \
    --vl_model_path /path/to/vl/model \
    --image_path image.png \
    --output_path output \
    --persistence_strategy latex

# 处理PDF
python main_refactored.py \
    --vl_model_path /path/to/vl/model \
    --pdf_path document.pdf \
    --output_path output \
    --persistence_strategy pdf
```

### 2. 编程使用

```python
from main_refactored import DocumentProcessor

# 创建文档处理器
processor = DocumentProcessor(
    vl_model_path="/path/to/vl/model",
    translator_model_path="/path/to/translation/model"
)

# 设置持久化策略
processor.set_persistence_strategy("latex")  # 或 "pdf"

# 处理文档
processor.process_image("image.png", "output")
processor.process_pdf("document.pdf", "output")
```

## 架构优势

1. **模块化**: 每个组件职责清晰，易于维护
2. **策略模式**: 输入转换器和持久化层都支持策略模式，易于扩展
3. **接口清晰**: 各组件间接口明确，便于测试和调试
4. **资源管理**: 自动管理模型加载和卸载，避免内存泄漏
5. **可扩展性**: 易于添加新的输入格式或输出格式

## 扩展指南

### 添加新的输入格式

1. 继承 `InputConverterStrategy`
2. 实现 `convert()` 方法
3. 在 `VLAdapter` 中使用新策略

### 添加新的输出格式

1. 继承 `PersistenceStrategy`
2. 实现 `save()` 方法
3. 在 `DocumentProcessor` 中添加新策略选项

### 实现PDF输出

在 `PdfPersistenceStrategy.save()` 方法中实现PDF生成逻辑，可以：
- 集成LaTeX编译
- 使用其他PDF生成库
- 调用外部PDF生成工具

## 注意事项

1. **模型路径**: 确保提供正确的模型路径
2. **设备配置**: 根据硬件配置调整设备参数
3. **内存管理**: 大模型会自动管理内存，但建议在处理完成后调用清理方法
4. **错误处理**: 各组件都有基本的错误处理，但建议在生产环境中加强
