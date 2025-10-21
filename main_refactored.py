"""
重构后的主程序 - 整合VLAdapter、LL和Persistence三个组件
"""
import argparse
import os
import re
from typing import List, Tuple

import cv2

from vladapter import VLAdapter, PictToPictConverter, PdfToPictConverter
from translator import TextTranslator
from persistence import PersistenceLayer, LatexPersistenceStrategy, PdfPersistenceStrategy, ImageProcessor


class DocumentProcessor:
    """文档处理器 - 整合所有组件"""
    
    def __init__(self, vl_model_path: str, translator_model_path: str = None, 
                 device: str = "cuda", attn: str = "sdpa", debug_content: bool = False):
        """
        初始化文档处理器
        
        Args:
            vl_model_path: 视觉模型路径
            translator_model_path: 翻译模型路径
            device: 设备类型
            attn: 注意力实现
        """
        # 初始化VLAdapter
        self.vl_adapter = VLAdapter(vl_model_path, device, attn)
        
        # 初始化LL（语言大模型）
        self.translator = TextTranslator(model_path=translator_model_path)
        
        # 初始化Persistence（持久化层）
        self.persistence = PersistenceLayer(LatexPersistenceStrategy())

        self.debug_content = debug_content
    
    def process_image(self, image_path: str, output_path: str, prompt: str = "QwenVL HTML") -> str:
        """
        处理单张图片
        
        Args:
            image_path: 图片路径
            output_path: 输出路径
            prompt: 提示词
            
        Returns:
            处理结果路径
        """
        print(f"开始处理图片: {image_path}")
        
        # 1. 设置输入转换器为图片到图片
        self.vl_adapter.set_input_converter(PictToPictConverter())
        
        # 2. 视觉推理
        html_contents, input_heights, input_widths = self.vl_adapter.process(image_path, prompt)
        
        # 3. 卸载视觉模型
        self.vl_adapter.cleanup()
        
        # 4. 处理图片并替换img标签
        processed_contents = self._process_images_and_replace_tags(
            [image_path], html_contents, output_path, input_heights, input_widths
        )

        # 5. 清理HTML标签
        cleaned_contents = [self.vl_adapter._clean_html_tags(process_content) for process_content in processed_contents]
        
        # 6. 翻译
        translated_contents = self._translate_contents(cleaned_contents)
        
        # 7. 持久化
        self.persistence.save(translated_contents, output_path)

        # 8. 输出调试内容
        self._output_debug_content(html_contents, output_path, [image_path], input_heights, input_widths, 1)
        
        print(f"图片处理完成: {output_path}")
        return output_path

    def _output_debug_content(self, html_contents: List[str], output_path: str, image_paths: List[str], input_heights: List[int], input_widths: List[int], start_page: int = 1):
        # 如果不需要调试内容, 直接返回
        if not self.debug_content:
            return
        # 获取输出dir
        output_dir = os.path.dirname(output_path)
        # 创建debug目录
        debug_dir = os.path.join(output_dir, "debug")
        os.makedirs(debug_dir, exist_ok=True)

        # 获取文件名
        filename = os.path.basename(output_path)
        # 获取文件名（不含扩展名）
        filename = filename.rsplit('.', 1)[0]
        """输出调试内容"""
        for i, data in enumerate(zip(html_contents, image_paths, input_heights, input_widths)):
            html_content, image_path, input_height, input_width = data
            # html
            file_path = os.path.join(debug_dir, f"{filename}_{start_page + i}_html.txt")
            with open(file_path, "w") as f:
                f.write(html_content)
            # 清理HTML标签
            file_path = os.path.join(debug_dir, f"{filename}_{start_page + i}_clear_div.txt")
            with open(file_path, "w") as f:
                f.write(self.vl_adapter._clean_html_tags(html_content))
            # 识别区间 
            file_path = os.path.join(debug_dir, f"{filename}_{start_page + i}_bbox.png")
            def plot_bbox(img_path, pred, input_height, input_width, output_path):
                img = cv2.imread(img_path)
                img_height, img_width, _ = img.shape
                scale = (img_width / input_width, img_height / input_height)
                bboxes = []

                pattern = re.compile(r'img data-bbox="(\d+),(\d+),(\d+),(\d+)"')

                scale_x, scale_y = scale  

                def replace_bbox(match):
                    x1, y1, x2, y2 = map(int, match)
                    bboxes.append([int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y)])

                
                matches = re.findall(pattern, pred)
                if matches:
                    for match in matches:
                        # print(match)
                        replace_bbox(match)
                for bbox in bboxes:
                    x1, y1, x2, y2 = bbox
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 8)

                cv2.imwrite(file_path, img)
            plot_bbox(image_path, html_content, input_height, input_width, file_path)
            
    def process_pdf(self, pdf_path: str, output_path: str, prompt: str = "QwenVL HTML", debug_content: bool = False) -> str:
        """
        处理PDF文件
        
        Args:
            pdf_path: PDF路径
            output_path: 输出路径
            prompt: 提示词
            
        Returns:
            处理结果路径
        """
        print(f"开始处理PDF: {pdf_path}")
        
        # 1. 设置输入转换器为PDF到图片
        pdf_converter = PdfToPictConverter()
        self.vl_adapter.set_input_converter(pdf_converter)
        
        try:
            # 2. 视觉推理
            html_contents, input_heights, input_widths = self.vl_adapter.process(pdf_path, prompt)
            
            # 3. 卸载视觉模型
            self.vl_adapter.cleanup()
            
            # 4. 获取图片路径用于处理
            image_paths = pdf_converter.convert(pdf_path)
            
            # 5. 处理图片并替换img标签
            processed_contents = self._process_images_and_replace_tags(
                image_paths, html_contents, output_path, input_heights, input_widths
            )

            # 6. 清理HTML标签
            cleaned_contents = [self.vl_adapter._clean_html_tags(process_content) for process_content in processed_contents]
            
            # 7. 翻译
            translated_contents = self._translate_contents(cleaned_contents)

            # 8. 持久化
            self.persistence.save(translated_contents, output_path)

            # 9. 输出调试内容
            self._output_debug_content(html_contents, output_path, image_paths, input_heights, input_widths)

            print(f"PDF处理完成: {output_path}")
            return output_path
            
        finally:
            # 清理临时图片
            pdf_converter.cleanup(image_paths)
    
    def _process_images_and_replace_tags(self, image_paths: List[str], html_contents: List[str], 
                                       output_path: str, input_heights: List[int], input_widths: List[int]) -> List[str]:
        """处理图片并替换img标签"""
        return ImageProcessor.process_images_and_replace_predictions(
            image_paths, html_contents, output_path, input_heights, input_widths
        )
    
    def _translate_contents(self, contents: List[str]) -> List[str]:
        """翻译内容"""
        try:
            print("开始翻译...")
            self.translator.load_model()
            translated_contents = []
            
            for i, content in enumerate(contents):
                print(f"翻译第{i+1}页...")
                translated = self.translator.translate(content)
                translated_contents.append(translated)
            
            return translated_contents
            
        except Exception as e:
            print(f"翻译失败: {e}")
            return contents  # 翻译失败时返回原文
        finally:
            self.translator.unload_model()
    
    def set_persistence_strategy(self, strategy_type: str):
        """设置持久化策略"""
        if strategy_type.lower() == "latex":
            self.persistence.set_strategy(LatexPersistenceStrategy())
        elif strategy_type.lower() == "pdf":
            self.persistence.set_strategy(PdfPersistenceStrategy())
        else:
            raise ValueError(f"不支持的持久化策略: {strategy_type}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="文档处理系统 - 支持图片和PDF处理")
    
    # 模型路径参数
    parser.add_argument("--vl_model_path", type=str, required=True,
                        help="视觉模型路径")
    parser.add_argument("--translator_model_path", type=str, default=None,
                        help="翻译模型路径（可选）")
    
    # 输入参数（互斥）
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--image_path", type=str,
                            help="输入图片路径")
    input_group.add_argument("--pdf_path", type=str,
                            help="输入PDF路径")
    
    # 输出参数
    parser.add_argument("--output_path", type=str, required=True,
                        help="输出路径")
    
    # 可选参数
    parser.add_argument("--prompt", type=str, default="QwenVL HTML",
                        help="提示词")
    parser.add_argument("--device", type=str, default="cuda",
                        help="设备类型")
    parser.add_argument("--attn", type=str, default="sdpa",
                        help="注意力实现")
    parser.add_argument("--persistence_strategy", type=str, default="latex",
                        choices=["latex", "pdf"],
                        help="持久化策略")
    parser.add_argument("--debug_content", type=bool, default=False,
                        help="是否输出调试内容")
    
    args = parser.parse_args()
    
    # 创建文档处理器
    processor = DocumentProcessor(
        vl_model_path=args.vl_model_path,
        translator_model_path=args.translator_model_path,
        device=args.device,
        attn=args.attn,
        debug_content=args.debug_content
    )
    
    # 设置持久化策略
    processor.set_persistence_strategy(args.persistence_strategy)
    
    # 处理文档
    if args.image_path:
        processor.process_image(args.image_path, args.output_path, args.prompt)
    else:
        processor.process_pdf(args.pdf_path, args.output_path, args.prompt)


if __name__ == "__main__":
    main()
