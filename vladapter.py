"""
VLAdapter - 视觉大模型适配器
包含输入转换器和视觉推理模型两个核心组件
"""
import torch
import os
import re
import math
import cv2
from abc import ABC, abstractmethod
from typing import List, Tuple, Union
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from PIL import Image
from pdf_to_image import PDFToImageConverter


class InputConverterStrategy(ABC):
    """输入转换器策略基类"""
    
    @abstractmethod
    def convert(self, input_path: str) -> List[str]:
        """
        转换输入文件为图片路径列表
        
        Args:
            input_path: 输入文件路径
            
        Returns:
            图片路径列表
        """
        pass


class PictToPictConverter(InputConverterStrategy):
    """图片到图片转换器（默认策略，不做任何转换）"""
    
    def convert(self, input_path: str) -> List[str]:
        """直接返回图片路径"""
        return [input_path]


class PdfToPictConverter(InputConverterStrategy):
    """PDF到图片转换器"""
    
    def __init__(self):
        self.converter = PDFToImageConverter()
    
    def convert(self, input_path: str) -> List[str]:
        """将PDF转换为图片路径列表"""
        return self.converter.convert_pdf_to_images_temp(input_path)
    
    def cleanup(self, image_paths: List[str]):
        """清理临时图片文件"""
        self.converter.cleanup_temp_images(image_paths)


class VisionModel:
    """视觉推理模型"""
    
    def __init__(self, model_path: str, device: str = "cuda", attn: str = "sdpa"):
        self.model_path = model_path
        self.device = device
        self.attn = attn
        self.model = None
        self.processor = None
        self._is_loaded = False
    
    def load_model(self):
        """加载模型和处理器"""
        if self._is_loaded:
            return
            
        device_map = "cuda" if self.device == "cuda" else "cpu"
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path, 
            torch_dtype=torch.bfloat16, 
            attn_implementation=self.attn, 
            device_map=device_map
        )
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self._is_loaded = True
        print("视觉模型加载完成")
    
    def translate(self, pict_path: str, prompt: str = "QwenVL HTML", 
                  system_prompt: str = "You are a helpful assistant") -> Tuple[List[str], int, int]:
        """
        翻译图片内容
        
        Args:
            pict_path: 图片路径
            prompt: 提示词
            system_prompt: 系统提示词
            
        Returns:
            (包含HTML标签的内容列表, 去除HTML标签的内容列表, 输入高度, 输入宽度)
        """
        if not self._is_loaded:
            self.load_model()
        
        print("开始推理...")
        image = Image.open(pict_path)
        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "image": pict_path
                    }
                ]
            }
        ]
        
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], padding=True, return_tensors="pt").to(self.model.device)
        
        print("准备生成...")
        output_ids = self.model.generate(**inputs, max_new_tokens=8192)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
        output_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        
        input_height = inputs['image_grid_thw'][0][1] * 14
        input_width = inputs['image_grid_thw'][0][2] * 14
        
        print("生成完成...")
        
        # 处理HTML内容
        html_content = output_text[0]
        
        return [html_content], input_height, input_width
        
    def unload_model(self):
        """卸载模型，释放显存"""
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        self._is_loaded = False
        torch.cuda.empty_cache()
        print("视觉模型已卸载，显存已释放")


class VLAdapter:
    """视觉大模型适配器"""
    
    def __init__(self, model_path: str, device: str = "cuda", attn: str = "sdpa"):
        self.model_path = model_path
        self.device = device
        self.attn = attn
        
        # 初始化组件
        self.input_converter: InputConverterStrategy = PictToPictConverter()  # 默认策略
        self.vision_model = VisionModel(model_path, device, attn)
    
    def set_input_converter(self, converter: InputConverterStrategy):
        """设置输入转换器策略"""
        self.input_converter = converter
    
    def process(self, input_path: str, prompt: str = "QwenVL HTML") -> Tuple[List[str], List[int], List[int]]:
        """
        处理输入文件
        
        Args:
            input_path: 输入文件路径
            prompt: 提示词
            
        Returns:
            (包含HTML标签的内容列表, 去除HTML标签的内容列表, 输入高度列表, 输入宽度列表)
        """
        # 1. 输入转换
        image_paths = self.input_converter.convert(input_path)
        
        # 2. 视觉推理
        html_contents = []
        input_heights = []
        input_widths = []
        
        for i, image_path in enumerate(image_paths):
            print(f"处理第{i+1}页...")
            html_content, input_height, input_width = self.vision_model.translate(image_path, prompt)
            html_contents.extend(html_content)
            input_heights.append(input_height)
            input_widths.append(input_width)
            print(f"处理第{i+1}页完成...")
        
        return html_contents, input_heights, input_widths

    def _clean_html_tags(self, html_content: str) -> str:
        """清理HTML标签"""
        output = html_content
        
        # 移除img标签
        IMG_RE = re.compile(
            r'<img\b[^>]*\bdata-bbox\s*=\s*"?\d+,\d+,\d+,\d+"?[^>]*\/?>',
            flags=re.IGNORECASE,
        )
        output = IMG_RE.sub('', output)
        
        # 处理p标签
        output = re.sub(
            r'<p\b[^>]*>(.*?)<\/p>',
            r'\1\n',
            output,
            flags=re.DOTALL | re.IGNORECASE,
        )
        
        # 处理div标签
        def strip_div(class_name: str, txt: str) -> str:
            pattern = re.compile(
                rf'\s*<div\b[^>]*class="{class_name}"[^>]*>(.*?)<\/div>\s*',
                flags=re.DOTALL | re.IGNORECASE,
            )
            if class_name == 'image':
                return pattern.sub(r' \n\n\1\n\n ', txt)
            return pattern.sub(r' \1 ', txt)
        
        for cls in ['image', 'chemistry', 'table', 'formula', 'image caption']:
            output = strip_div(cls, output)
        
        output = output.replace(" </td>", "</td>")
        return output
   
    def cleanup(self):
        """清理资源"""
        self.vision_model.unload_model()
        if hasattr(self.input_converter, 'cleanup'):
            # 如果有临时文件需要清理
            pass
