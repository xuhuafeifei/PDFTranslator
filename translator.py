from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

import logging

class TextTranslator:
    """
    文本翻译器类
    支持通过输入字符串进行翻译，返回翻译结果
    """
    
    def __init__(self, model_path=None, model_name="Qwen/Qwen2.5-7.5B-Instruct", device="auto"):
        """
        初始化翻译器
        
        参数:
            model_path: 本地模型路径（优先使用）
            model_name: 模型名称（当model_path为None时使用）
            device: 设备类型
        """
        self.model_path = model_path
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None
        self._is_loaded = False
        
        # 设置日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def load_model(self):
        """加载模型和分词器"""
        if self._is_loaded:
            self.logger.info("模型已经加载，跳过重复加载")
            return
            
        try:
            # 确定使用的模型路径
            if self.model_path:
                model_to_load = self.model_path
                self.logger.info(f"正在加载本地模型: {self.model_path}")
            else:
                model_to_load = self.model_name
                self.logger.info(f"正在加载在线模型: {self.model_name}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_to_load, 
                trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_to_load,
                torch_dtype=torch.float16,
                device_map=self.device,
                trust_remote_code=True
            )
            self._is_loaded = True
            self.logger.info("模型加载完成")
        except Exception as e:
            self.logger.error(f"模型加载失败: {e}")
            raise
    
    def translate(self, text, source_lang="英文", target_lang="中文", preserve_format=True, max_tokens=2048):
        """
        翻译文本
        
        参数:
            text: 要翻译的文本
            source_lang: 源语言
            target_lang: 目标语言
            preserve_format: 是否保持格式（LaTeX命令等）
            max_tokens: 最大生成token数，None时根据文本长度自动调整
        
        返回:
            翻译后的文本
        """
        if not self._is_loaded:
            self.load_model()
        
        if not text or not text.strip():
            self.logger.warning("输入文本为空")
            return text
        
        try:
            # 构建翻译提示词
            if preserve_format:
                prompt = f"""请将以下{source_lang}翻译成{target_lang}，要求：
1. 保持所有LaTeX命令不变（如 $, \\, \\begin, \\end, \\section, \\title, \\includegraphics等）
2. 保持所有LaTeX环境不变（如 \\begin{{center}}, \\begin{{figure}}等）
3. 保持所有数学公式不变（如 $\\alpha$, $$...$$等）
4. 保持所有换行符和段落结构不变
5. 只翻译纯文本内容，不要添加任何解释或注释, 这点最重要

原文：
{text}
"""
            else:
                prompt = f"请将以下{source_lang}翻译成{target_lang}：\n\n{text}\n\n"
            
            self.logger.info(f"输入文本: {len(text)}字符, 设置max_new_tokens: {max_tokens}")
            
            # 执行翻译
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            outputs = self.model.generate(**inputs, max_new_tokens=max_tokens)
            translated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 提取翻译结果（去掉提示词部分）
            if "译文：" in translated:
                translated = translated.split("译文：")[-1].strip()
            
            self.logger.info(f"翻译完成，原文长度: {len(text)}, 译文长度: {len(translated)}, 使用max_tokens: {max_tokens}")
            return translated
            
        except Exception as e:
            self.logger.error(f"翻译失败: {e}")
            return text  # 翻译失败时返回原文
    
    def translate_batch(self, texts, source_lang="英文", target_lang="中文", preserve_format=True):
        """
        批量翻译文本
        
        参数:
            texts: 要翻译的文本列表
            source_lang: 源语言
            target_lang: 目标语言
            preserve_format: 是否保持格式
        
        返回:
            翻译后的文本列表
        """
        if not self._is_loaded:
            self.load_model()
        
        results = []
        for i, text in enumerate(texts):
            self.logger.info(f"正在翻译第 {i+1}/{len(texts)} 个文本")
            translated = self.translate(text, source_lang, target_lang, preserve_format)
            results.append(translated)
        
        return results
    
    def unload_model(self):
        """卸载模型，释放内存"""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        self._is_loaded = False
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        self.logger.info("模型已卸载")

# 使用示例
if __name__ == "__main__":
    import argparse
    
    # 命令行参数解析
    parser = argparse.ArgumentParser(description="文本翻译器")
    parser.add_argument("--model_path", type=str, help="本地模型路径")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct", help="在线模型名称")
    parser.add_argument("--device", type=str, default="auto", help="设备类型")
    parser.add_argument("--text", type=str, help="要翻译的文本")
    
    args = parser.parse_args()
    
    # 创建翻译器实例
    translator = TextTranslator()
    
    # 示例文本
    sample_text = """
    \\section{Introduction}
    This is a sample text for translation.
    The model should preserve LaTeX commands like $\\alpha$ and \\begin{equation}.
    """
    
    # 执行翻译
    translated_text = translator.translate(sample_text)
    print("翻译结果:")
    print(translated_text)
    
    # 卸载模型
    translator.unload_model()