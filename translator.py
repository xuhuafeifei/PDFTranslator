from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

import logging

class TextTranslator:
    """
    文本翻译器类
    支持通过输入字符串进行翻译，返回翻译结果
    """
    
    def __init__(self, model_path=None, model_name="Qwen/Qwen3-4B-Instruct-2507", device="cuda"):
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
            
            self.tokenizer = tokenizer = AutoTokenizer.from_pretrained(model_to_load)

            self.model = AutoModelForCausalLM.from_pretrained(
                model_to_load,
                torch_dtype="auto",
                device_map="auto"
            )
            self._is_loaded = True
            self.logger.info("模型加载完成")
        except Exception as e:
            self.logger.error(f"模型加载失败: {e}")
            raise
    
    def translate(self, text, source_lang="英文", target_lang="中文", preserve_format=True, max_tokens=2048, output_log: bool = True):
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
            self.logger.info(f"输入文本: {len(text)}字符, 设置max_new_tokens: {max_tokens}")

            user_prompt = f'''请将以下 LaTeX 文本从英文翻译成中文，严格遵循以下要求：
1. 保留所有 LaTeX 命令及参数不变,（例如 \\title{{content}}, \\author, \\begin, \\end, \\section, \\includegraphics）。
2. 保留所有 LaTeX 环境结构不变（例如 \\begin{{abstract}}, \\begin{{figure}}, \\begin{{center}}, \\title{{content}} 等）。
3. 保留所有数学公式原样（包括 $...$, \[...\], \( ... \), \\begin{{equation}} 等）。
4. 翻译后文本应适合直接用于 LaTeX 文档。
原文：
{text}

译文：
'''
            # 准备model input
            messages = [
                {"role": "system", "content": "你是一个 LaTeX 翻译器，只输出翻译后的 LaTeX，不输出任何解释。"},
                {"role": "user", "content": user_prompt}
            ]
            inputs = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            model_inputs = self.tokenizer([inputs], return_tensors="pt").to(self.model.device)
            
            # 执行翻译
            generated_ids = self.model.generate(
                **model_inputs, 
                max_new_tokens=max_tokens
            )
            output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

            translated = self.tokenizer.decode(output_ids, skip_special_tokens=True)     

            # 提取翻译结果（去掉提示词部分）
            if "译文：" in translated:
                translated = translated.split("译文：")[-1].strip()
            

            def remove_first_last_lines(text):
                """去除第一行和最后一行"""
                lines = text.splitlines()
                if len(lines) <= 2:
                    return ""  # 如果只有2行或更少，返回空字符串
                return '\n'.join(lines[1:-1])

            # translated =  remove_first_last_lines(translated)
            if output_log:
                self.logger.info(f"翻译完成，原文长度: {len(text)}, 译文长度: {len(translated)}, 使用max_tokens: {max_tokens}\n翻译内容: {translated}")

            return translated
            
        except Exception as e:
            self.logger.error(f"翻译失败: {e}")
            return text  # 翻译失败时返回原文
    
    def translate_optimized(self, text, source_lang="英文", target_lang="中文", preserve_format=True, max_tokens=2048):
        # 切割text
        lines = text.strip().split("\n")
        result = []
        for line in lines:
            if line.strip() == "":
                result.append("\n")
            else:
                translated = self.translate(line, source_lang, target_lang, preserve_format, max_tokens, output_log=False)
                result.append(translated)
        translated = "\n".join(result)
        self.logger.info(f"翻译完成，原文长度: {len(text)}, 译文长度: {len(translated)}, 使用max_tokens: {max_tokens}\n翻译内容: {translated}")
        return translated

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
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-VL-8B-Instruct-FP8", help="在线模型名称")
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