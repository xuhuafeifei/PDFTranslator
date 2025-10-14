import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

class LaTeXTranslator:
    def __init__(self, model_path=None, model_name="facebook/mbart-large-50-many-to-many-mmt", device="auto"):
        self.device = device if device != "auto" else "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.model_path = model_path

        # 加载 tokenizer 和 model
        model_to_load = model_path if model_path else model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_to_load)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_to_load).to(self.device)

    def _extract_latex(self, text):
        """
        找出LaTeX公式或命令，并替换成占位符
        支持：
          1. \begin{...} ... \end{...} 环境
          2. $...$ 行内公式
          3. \命令，带可选参数 [...] 和必选参数 {...}
        """
        pattern = r"""
            (\\begin\{.*?\}.*?\\end\{.*?\})    # 环境
            | (\$.*?\$)                         # 行内公式
            | (\\[a-zA-Z]+\*?(?:\[[^\]]*\])?(?:\{[^\}]*\})*)  # 命令及参数
        """
        self.latex_map = {}
        def repl(match):
            idx = len(self.latex_map)
            key = f"[[CMD{idx}]]"
            self.latex_map[key] = match.group(0)
            return key

        text_masked = re.sub(pattern, repl, text, flags=re.DOTALL | re.VERBOSE)
        return text_masked

    def _restore_latex(self, text):
        """将占位符恢复为原始LaTeX"""
        for key, val in self.latex_map.items():
            text = text.replace(key, val)
        return text

    def translate(self, text, source_lang="en", target_lang="zh"):
        """翻译文本，同时保护LaTeX"""
        # 1. 替换 LaTeX
        text_masked = self._extract_latex(text)

        # 2. 翻译
        inputs = self.tokenizer(text_masked, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**inputs, max_new_tokens=1024)
        translated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 3. 恢复 LaTeX
        final_text = self._restore_latex(translated)
        return final_text


if __name__ == "__main__":
    translator = LaTeXTranslator()

    sample_text = r"""
    \title{虚拟时间与分布式系统的全局状态}
    \author{Friedemann Mattern}
    \section{Introduction}
    This is a sample text for translation. The formula is $E = mc^2$.
    \begin{equation}
    F = ma
    \end{equation}
    """

    translated = translator.translate(sample_text)
    print("翻译结果:")
    print(translated)
