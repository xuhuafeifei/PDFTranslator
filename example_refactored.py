"""
使用示例 - 展示重构后的代码如何使用
"""
from vladapter import VLAdapter, PictToPictConverter, PdfToPictConverter
from translator import TextTranslator
from persistence import PersistenceLayer, LatexPersistenceStrategy, PdfPersistenceStrategy


def example_image_processing():
    """图片处理示例"""
    print("=== 图片处理示例 ===")
    
    # 1. 创建VLAdapter
    vl_adapter = VLAdapter(
        model_path="/path/to/your/vl/model",
        device="cuda",
        attn="sdpa"
    )
    
    # 2. 设置输入转换器（图片到图片）
    vl_adapter.set_input_converter(PictToPictConverter())
    
    # 3. 处理图片
    image_path = "example.png"
    html_contents, clean_contents, input_heights, input_widths = vl_adapter.process(image_path)
    
    # 4. 清理资源
    vl_adapter.cleanup()
    
    print(f"处理完成，生成了 {len(html_contents)} 个内容")


def example_pdf_processing():
    """PDF处理示例"""
    print("=== PDF处理示例 ===")
    
    # 1. 创建VLAdapter
    vl_adapter = VLAdapter(
        model_path="/path/to/your/vl/model",
        device="cuda",
        attn="sdpa"
    )
    
    # 2. 设置输入转换器（PDF到图片）
    pdf_converter = PdfToPictConverter()
    vl_adapter.set_input_converter(pdf_converter)
    
    try:
        # 3. 处理PDF
        pdf_path = "example.pdf"
        html_contents, clean_contents, input_heights, input_widths = vl_adapter.process(pdf_path)
        
        # 4. 获取图片路径
        image_paths = pdf_converter.convert(pdf_path)
        
        print(f"PDF处理完成，共 {len(image_paths)} 页")
        
    finally:
        # 5. 清理资源
        vl_adapter.cleanup()
        pdf_converter.cleanup(image_paths)


def example_translation():
    """翻译示例"""
    print("=== 翻译示例 ===")
    
    # 1. 创建翻译器
    translator = TextTranslator(
        model_path="/path/to/your/translation/model"
    )
    
    # 2. 翻译文本
    text = "\\section{Introduction}\nThis is a sample text."
    translated = translator.translate(text)
    
    print(f"原文: {text}")
    print(f"译文: {translated}")
    
    # 3. 清理资源
    translator.unload_model()


def example_persistence():
    """持久化示例"""
    print("=== 持久化示例 ===")
    
    # 1. 创建持久化层（LaTeX策略）
    persistence = PersistenceLayer(LatexPersistenceStrategy())
    
    # 2. 保存内容
    content = ["\\section{Introduction}", "This is content."]
    output_path = "output"
    persistence.save(content, output_path)
    
    print("LaTeX文件已保存")
    
    # 3. 切换到PDF策略
    persistence.set_strategy(PdfPersistenceStrategy())
    persistence.save(content, output_path)
    
    print("PDF策略已设置（空壳实现）")


def example_complete_workflow():
    """完整工作流示例"""
    print("=== 完整工作流示例 ===")
    
    # 这个示例展示了如何使用重构后的代码进行完整的文档处理
    # 实际使用时，建议使用 main_refactored.py 中的 DocumentProcessor 类
    
    from main_refactored import DocumentProcessor
    
    # 创建文档处理器
    processor = DocumentProcessor(
        vl_model_path="/path/to/your/vl/model",
        translator_model_path="/path/to/your/translation/model",
        device="cuda",
        attn="sdpa"
    )
    
    # 设置持久化策略
    processor.set_persistence_strategy("latex")
    
    # 处理图片
    processor.process_image("example.png", "output_image")
    
    # 处理PDF
    processor.process_pdf("example.pdf", "output_pdf")


if __name__ == "__main__":
    print("重构后的代码使用示例")
    print("注意：这些示例需要实际的模型路径才能运行")
    
    # 运行示例（需要实际的模型路径）
    # example_image_processing()
    # example_pdf_processing()
    # example_translation()
    # example_persistence()
    # example_complete_workflow()
    
    print("\n要运行实际的处理，请使用:")
    print("python main_refactored.py --vl_model_path /path/to/vl/model --image_path image.png --output_path output")
