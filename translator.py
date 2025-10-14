from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import logging

class TextTranslator:
    """
    文本翻译器类
    支持通过输入字符串进行翻译，返回翻译结果
    """
    
    def __init__(self, model_name="Qwen/Qwen2.5-1.5B-Instruct", device="auto"):
        """
        初始化翻译器
        
        参数:
            model_name: 模型名称
            device: 设备类型
        """
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
            self.logger.info(f"正在加载模型: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, 
                trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map=self.device,
                trust_remote_code=True
            )
            self._is_loaded = True
            self.logger.info("模型加载完成")
        except Exception as e:
            self.logger.error(f"模型加载失败: {e}")
            raise
    
    def translate(self, text, source_lang="英文", target_lang="中文", preserve_format=True):
        """
        翻译文本
        
        参数:
            text: 要翻译的文本
            source_lang: 源语言
            target_lang: 目标语言
            preserve_format: 是否保持格式（LaTeX命令等）
        
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
                prompt = f"请将以下{source_lang}翻译成{target_lang}，并保持所有 LaTeX 命令（如 $, \\ , \\begin 等）不变：\n\n{text}\n\n译文："
            else:
                prompt = f"请将以下{source_lang}翻译成{target_lang}：\n\n{text}\n\n译文："
            
            # 执行翻译
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            outputs = self.model.generate(**inputs, max_new_tokens=2048)
            translated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 提取翻译结果（去掉提示词部分）
            if "译文：" in translated:
                translated = translated.split("译文：")[-1].strip()
            
            self.logger.info(f"翻译完成，原文长度: {len(text)}, 译文长度: {len(translated)}")
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
