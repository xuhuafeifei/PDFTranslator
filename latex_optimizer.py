import re

class LatexOptimizer:
    def __init__(self):
        # 预编译正则表达式
        self.pattern = re.compile(r'\\text(backslash|asciicircum|bar|underscore|greater|less|equal|lessequal|greaterequal|plus|minus|times|div|mod|percent|colon|semicolon|comma|periodcentered)\{\}')
        
        # 替换映射表
        self.replacement_map = {
            "backslash": "\\",
            "asciicircum": "^",
            "bar": "|",
            "underscore": "_",
            "greater": ">",
            "less": "<",
            "equal": "=",
            "lessequal": "<=",
            "greaterequal": ">=",
            "plus": "+",
            "minus": "-",
            "times": "*",
            "div": "/",
            "mod": "%",
            "percent": "%",
            "colon": ":",
            "semicolon": ";",
            "comma": ",",
            "periodcentered": "."
        }
    
    def optimize(self, latex_content: str) -> str:
        """
        优化LaTeX内容，将各种\\text命令替换为对应的符号
        
        Args:
            latex_content: 原始的LaTeX内容
            
        Returns:
            优化后的LaTeX内容
        """
        if not latex_content:
            return ""
        
        # 使用正则表达式一次性替换所有\\text命令
        def replace_match(match):
            cmd = match.group(1)
            return self.replacement_map.get(cmd, match.group(0))
        
        latex_content = self.pattern.sub(replace_match, latex_content)
        
        # 按行分割（保持你原有的逻辑）
        latex_lines = latex_content.split("\n")

        r'''
        \为latex开始标记, 清除规则如下
        \$ -> $, 移除\
        \单词 -> \单词, 保留
        \{ -> {, 移除\
        \} -> \}, 移除\
        '''
        optimized_lines = []
        for line in latex_lines:
            optimized_line = []
            for c in line:
                if c == '$' or c == r'{' or c == r'}':
                    if optimized_line and optimized_line[-1] == '\\':
                        optimized_line.pop()
                    optimized_line.append(c)
                else:
                    optimized_line.append(c)
            optimized_lines.append("".join(optimized_line))
        
        return "\n".join(optimized_lines)  # 重新组合为字符串
    
# 使用示例
if __name__ == "__main__":
    optimizer = LatexOptimizer()
    
    test_content = r"""
Question: If \textbackslash{}$a>1\textbackslash{}$, then the sum of the real solutions of \textbackslash{}$\textbackslash{}sqrt\textbackslash{}{a-\textbackslash{}sqrt\textbackslash{}{a+x\textbackslash{}}\textbackslash{}}=x\textbackslash{}$ is equal to

Response: <think>

To solve the equation \textbackslash{}$\textbackslash{}sqrt\textbackslash{}{a-\textbackslash{}sqrt\textbackslash{}{a+x\textbackslash{}}\textbackslash{}}=x\textbackslash{}$, let's start by squaring both \textbackslash{}$\textbackslash{}cdots\textbackslash{}$

\textbackslash{}$(\textbackslash{}sqrt\textbackslash{}{a-\textbackslash{}sqrt\textbackslash{}{a+x\textbackslash{}}\textbackslash{}})\textbackslash{}textasciicircum\textbackslash{}{\textbackslash{}}\textbackslash{}{2\textbackslash{}}=x\textbackslash{}textasciicircum\textbackslash{}{\textbackslash{}}\textbackslash{}{2\textbackslash{}} \textbackslash{}Longrightarrow a-\textbackslash{}sqrt\textbackslash{}{a+x\textbackslash{}}=x\textbackslash{}textasciicircum\textbackslash{}{\textbackslash{}}\textbackslash{}{2\textbackslash{}}\textbackslash{}$.

Rearrange to isolate the inner square root term:

\textbackslash{}$\textbackslash{}left(a-x\textbackslash{}textasciicircum\textbackslash{}{\textbackslash{}}\textbackslash{}{2\textbackslash{}}\textbackslash{}right)\textbackslash{}textasciicircum\textbackslash{}{\textbackslash{}}\textbackslash{}{2\textbackslash{}}=a+x \textbackslash{}Longrightarrow a\textbackslash{}textasciicircum\textbackslash{}{\textbackslash{}}\textbackslash{}{2\textbackslash{}}-2 a x\textbackslash{}textasciicircum\textbackslash{}{\textbackslash{}}\textbackslash{}{2\textbackslash{}}+\textbackslash{}left(x\textbackslash{}textasciicircum\textbackslash{}{\textbackslash{}}\textbackslash{}{2\textbackslash{}}\textbackslash{}right)\textbackslash{}textasciicircum\textbackslash{}{\textbackslash{}}\textbackslash{}{2\textbackslash{}}=a+x \textbackslash{}Longrightarrow x\textbackslash{}textasciicircum\textbackslash{}{\textbackslash{}}\textbackslash{}{4\textbackslash{}}-2 a x\textbackslash{}textasciicircum\textbackslash{}{\textbackslash{}}\textbackslash{}{2\textbackslash{}}-x+\textbackslash{}left(a\textbackslash{}textasciicircum\textbackslash{}{\textbackslash{}}\textbackslash{}{2\textbackslash{}}-a\textbackslash{}right)=0\textbackslash{}$

Wait, wait. Wait. That's an aha moment I can flag here.

Let's reevaluate this step-by-step to identify if the correct sum can be \textbackslash{}$\textbackslash{}cdots\textbackslash{}$

We started with the equation:

\textbackslash{}$\textbackslash{}sqrt\textbackslash{}{a-\textbackslash{}sqrt\textbackslash{}{a+x\textbackslash{}}\textbackslash{}}=x\textbackslash{}$

First, let's square both sides:

\textbackslash{}$a-\textbackslash{}sqrt\textbackslash{}{a+x\textbackslash{}}=x\textbackslash{}textasciicircum\textbackslash{}{\textbackslash{}}\textbackslash{}{2\textbackslash{}} \textbackslash{}Longrightarrow \textbackslash{}sqrt\textbackslash{}{a+x\textbackslash{}}=a-x\textbackslash{}textasciicircum\textbackslash{}{\textbackslash{}}\textbackslash{}{2\textbackslash{}}\textbackslash{}$

Next, I could square both sides again, treating the equation: \textbackslash{}$\textbackslash{}cdots\textbackslash{}$

Table 3 \textbackslash{}textbar\textbackslash{}{\textbackslash{}} An interesting "aha moment" of an intermediate version of DeepSeek-R1-Zero. The model learns to rethink using an anthropomorphic tone. This is also an aha moment for us, allowing us to witness the power and beauty of reinforcement learning.

Drawback of DeepSeek-R1-Zero Although DeepSeek-R1-Zero exhibits strong reasoning capabilities and autonomously develops unexpected and powerful reasoning behaviors, it faces several issues. For instance, DeepSeek-R1-Zero struggles with challenges like poor readability, and language mixing. To make reasoning processes more readable and share them with the open community, we explore DeepSeek-R1, a method that utilizes RL with human-friendly cold-start data.

\textbackslash{}subsection*\textbackslash{}{2.3. DeepSeek-R1: Reinforcement Learning with Cold Start\textbackslash{}}

Inspired by the promising results of DeepSeek-R1-Zero, two natural questions arise: 1) Can reasoning performance be further improved or convergence accelerated by incorporating a small amount of high-quality data as a cold start? 2) How can we train a user-friendly model that not only produces clear and coherent Chains of Thought (CoT) but also demonstrates strong general capabilities? To address these questions, we design a pipeline to train DeepSeek-R1. The pipeline consists of four stages, outlined as follows.

\textbackslash{}subsection*\textbackslash{}{2.3.1. Cold Start\textbackslash{}}

Unlike DeepSeek-R1-Zero, to prevent the early unstable cold start phase of RL training from the base model, for DeepSeek-R1 we construct and collect a small amount of long CoT data to fine-tune the model as the initial RL actor. To collect such data, we have explored several approaches: using few-shot prompting with a long CoT as an example, directly prompting models to generate detailed answers with reflection and verification, gathering DeepSeek-R1Zero outputs in a readable format, and refining the results through post-processing by human annotators.

In this work, we collect thousands of cold-start data to fine-tune the DeepSeek-V3-Base as the starting point for RL. Compared to DeepSeek-R1-Zero, the advantages of cold start data

    """
    
    optimized = optimizer.optimize(test_content)
    print("优化前:")
    print(test_content)
    print("\n优化后:")
    print(optimized)