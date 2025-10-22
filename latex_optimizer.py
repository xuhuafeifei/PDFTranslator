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
        # 特殊字符映射
        self.special_map = {
            # '\\': r'\textbackslash{}',
            # '{': r'\{',
            # '}': r'\}',
            # '$': r'\$',
            # '%': r'\%',
            '#': r'\#',
            '_': r'\_',
            # '^': r'\^{}',
            # '&': r'\&',
            # '~': r'\textasciitilde{}',
            '<': r'\textless{}',
            '>': r'\textgreater{}',
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

        def list_end_with(list, target) -> bool:
            if not list:
                return False
            if len(list) < len(target):
                return False
            return "".join(list[-len(target):]) == target

        in_math_mode = False

        for line in latex_lines:
            optimized_line = []
            for c in line:
                if c == '$' or c == '\\':
                    if c == '$':
                        in_math_mode = not in_math_mode
                    if optimized_line and optimized_line[-1] == '\\':
                        optimized_line.pop()
                    optimized_line.append(c)
                elif c == r'{':
                    if optimized_line and optimized_line[-1] == '\\':
                        optimized_line.pop()
                        # 判断结尾是否是'left', 如果是, 需要保留\
                        if list_end_with(optimized_line, 'left'):
                            optimized_line.append('\\')
                    optimized_line.append(c)
                elif c == r'}':
                    if optimized_line and optimized_line[-1] == '\\':
                        optimized_line.pop()
                        # 判断结尾是否是right, 如果是, 需要保留\
                        if list_end_with(optimized_line, 'right'):
                            optimized_line.append('\\')
                    optimized_line.append(c)
                # 非数学模式的文本, 需要将特殊字符映射转换
                elif not in_math_mode and c in self.special_map:
                    optimized_line.append(self.special_map[c])
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

#     test_content = r"""
# \textbackslash{}subsection*\textbackslash{}{3.1. DeepSeek-R1 Evaluation\textbackslash{}} <table><tr><td>Benchmark (Metric)</td><td>Claude-3.5-Sonnet-1022</td><td>GPT-4o 0513</td><td>DeepSeek V3</td><td>OpenAI o1-mini</td><td>OpenAI o1-1217</td><td>DeepSeek R1</td></tr><tr><td>Architecture</td><td>-</td><td>-</td><td>MoE</td><td>-</td><td>-</td><td>MoE</td></tr><tr><td>\textbackslash{}# Activated Params</td><td>-</td><td>-</td><td>37B</td><td>-</td><td>-</td><td>37B</td></tr><tr><td>\textbackslash{}# Total Params</td><td>-</td><td>-</td><td>671B</td><td>-</td><td>-</td><td>671B</td></tr><tr><td rowspan="9">English</td><td>MMLU (Pass@1)</td><td>88.3</td><td>87.2</td><td>88.5</td><td>85.2</td><td>91.8</td><td>90.8</td></tr><tr><td>MMLU-Redux (EM)</td><td>88.9</td><td>88.0</td><td>89.1</td><td>86.7</td><td>-</td><td>92.9</td></tr><tr><td>MMLU-Pro (EM)</td><td>78.0</td><td>72.6</td><td>75.9</td><td>80.3</td><td>-</td><td>84.0</td></tr><tr><td>DROP (3-shot FI)</td><td>88.3</td><td>83.7</td><td>91.6</td><td>83.9</td><td>90.2</td><td>92.2</td></tr><tr><td>IF-Eval (Prompt Strict)</td><td>86.5</td><td>84.3</td><td>86.1</td><td>84.8</td><td>-</td><td>83.3</td></tr><tr><td>GPQA Diamond (Pass@1)</td><td>65.0</td><td>49.9</td><td>59.1</td><td>60.0</td><td>75.7</td><td>71.5</td></tr><tr><td>SimpleQA (Correct)</td><td>28.4</td><td>38.2</td><td>24.9</td><td>7.0</td><td>47.0</td><td>30.1</td></tr><tr><td>FRAMES (Acc.)</td><td>72.5</td><td>80.5</td><td>73.3</td><td>76.9</td><td>-</td><td>82.5</td></tr><tr><td>AlpacaEval2.0 (LC-winrate)</td><td>52.0</td><td>51.1</td><td>70.0</td><td>57.8</td><td>-</td><td>87.6</td></tr><tr><td>ArenaHard (GPT-4-1106)</td><td>85.2</td><td>80.4</td><td>85.5</td><td>92.0</td><td>-</td><td>92.3</td></tr><tr><td rowspan="5">Code</td><td>LiveCodeBench (Pass@1-COT)</td><td>38.9</td><td>32.9</td><td>36.2</td><td>53.8</td><td>63.4</td><td>65.9</td></tr><tr><td>Codeforces (Percentile)</td><td>20.3</td><td>23.6</td><td>58.7</td><td>93.4</td><td>96.6</td><td>96.3</td></tr><tr><td>Codeforces (Rating)</td><td>717</td><td>759</td><td>1134</td><td>1820</td><td>2061</td><td>2029</td></tr><tr><td>SWE Verified (Resolved)</td><td>50.8</td><td>38.8</td><td>42.0</td><td>41.6</td><td>48.9</td><td>49.2</td></tr><tr><td>Aider-Polyglot (Acc.)</td><td>45.3</td><td>16.0</td><td>49.6</td><td>32.9</td><td>61.7</td><td>53.3</td></tr><tr><td rowspan="3">Math</td><td>AIME 2024 (Pass@1)</td><td>16.0</td><td>9.3</td><td>39.2</td><td>63.6</td><td>79.2</td><td>79.8</td></tr><tr><td>MATH-500 (Pass@1)</td><td>78.3</td><td>74.6</td><td>90.2</td><td>90.0</td><td>96.4</td><td>97.3</td></tr><tr><td>CNMO 2024 (Pass@1)</td><td>13.1</td><td>10.8</td><td>43.2</td><td>67.6</td><td>-</td><td>78.8</td></tr><tr><td rowspan="3">Chinese</td><td>CLUEWSC (EM)</td><td>85.4</td><td>87.9</td><td>90.9</td><td>89.9</td><td>-</td><td>92.8</td></tr><tr><td>C-Eval (EM)</td><td>76.7</td><td>76.0</td><td>86.5</td><td>68.9</td><td>-</td><td>91.8</td></tr><tr><td>C-SimpleQA (Correct)</td><td>55.4</td><td>58.7</td><td>68.0</td><td>40.3</td><td>-</td><td>63.7</td></tr></table> <div class="table caption" data-bbox="151,629,686,648">Table 4 \textbackslash{}textbar\textbackslash{}{\textbackslash{}} Comparison between DeepSeek-R1 and other representative models.</div>
# For education-oriented knowledge benchmarks such as MMLU, MMLU-Pro, and GPQA Diamond, DeepSeek-R1 demonstrates superior performance compared to DeepSeek-V3. This improvement is primarily attributed to enhanced accuracy in STEM-related questions, where significant gains are achieved through large-scale reinforcement learning. Additionally, DeepSeek-R1 excels on FRAMES, a long-context-dependent QA task, showcasing its strong document analysis capabilities. This highlights the potential of reasoning models in AI-driven search and data analysis tasks. On the factual benchmark SimpleQA, DeepSeek-R1 outperforms DeepSeek-V3, demonstrating its capability in handling fact-based queries. A similar trend is observed where OpenAI-o1 surpasses GPT-4o on this benchmark. However, DeepSeek-R1 performs worse than DeepSeek-V3 on the Chinese SimpleQA benchmark, primarily due to its tendency to refuse answering certain queries after safety RL. Without safety RL, DeepSeek-R1 could achieve an accuracy of over \textbackslash{}$70 \textbackslash{}\textbackslash{}%\textbackslash{}$.

# DeepSeek-R1 also delivers impressive results on IF-Eval, a benchmark designed to assess a model's ability to follow format instructions. These improvements can be linked to the inclusion of instruction-following data during the final stages of supervised fine-tuning (SFT) and RL training. Furthermore, remarkable performance is observed on AlpacaEval2.0 and ArenaHard, indicating DeepSeek-R1's strengths in writing tasks and open-domain question answering. Its significant outperformance of DeepSeek-V3 underscores the generalization benefits of large-scale RL, which not only boosts reasoning capabilities but also improves performance across diverse domains. Moreover, the summary lengths generated by DeepSeek-R1 are concise, with an average of 689 tokens on ArenaHard and 2,218 characters on AlpacaEval 2.0. This indicates that

# """

    test_content = r"""
    Question: If $a>1$, then the sum of the real solutions of $\sqrt{a-\sqrt{a+x}}=x$ is equal to

Response: <think>

To solve the equation $\sqrt{a-\sqrt{a+x}}=x$, let's start by squaring both $\cdots$

$(\sqrt{a-\sqrt{a+x}})^{2}=x^{2} \Longrightarrow a-\sqrt{a+x}=x^{2}$.

Rearrange to isolate the inner square root term:

$\left(a-x^{2}\right)^{2}=a+x \Longrightarrow a^{2}-2 a x^{2}+\left(x^{2}\right)^{2}=a+x \Longrightarrow x^{4}-2 a x^{2}-x+\left(a^{2}-a\right)=0$

Wait, wait. Wait. That's an aha moment I can flag her

Let's reevaluate this step-by-step to identify if the correct sum can be $\cdots$

We started with the equation:

$\sqrt{a-\sqrt{a+x}}=x$

First, let's square both sides:

$a-\sqrt{a+x}=x^{2} \Longrightarrow \sqrt{a+x}=a-x^{2}$

Next, I could square both sides again, treating the equation: $\cdots$

Table 3 | An interesting "aha moment" of an intermediate version of DeepSeek-R1-Zero. The model learns to rethink using an anthropomorphic tone. This is also an aha moment for us, allowing us to witness the power and beauty of reinforcement learning.

Drawback of DeepSeek-R1-Zero Although DeepSeek-R1-Zero exhibits strong reasoning capabilities and autonomously develops unexpected and powerful reasoning behaviors, it faces several issues. For instance, DeepSeek-R1-Zero struggles with challenges like poor readability, and language mixing. To make reasoning processes more readable and share them with the open community, we explore DeepSeek-R1, a method that utilizes RL with human-friendly cold-start data.

\subsection*{2.3. DeepSeek-R1: Reinforcement Learning with Cold Start}

Inspired by the promising results of DeepSeek-R1-Zero, two natural questions arise: 1) Can reasoning performance be further improved or convergence accelerated by incorporating a small amount of high-quality data as a cold start? 2) How can we train a user-friendly model that not only produces clear and coherent Chains of Thought (CoT) but also demonstrates strong general capabilities? To address these questions, we design a pipeline to train DeepSeek-R1. The pipeline consists of four stages, outlined as follows.

\subsection*{2.3.1. Cold Start}

Unlike DeepSeek-R1-Zero, to prevent the early unstable cold start phase of RL training from the base model, for DeepSeek-R1 we construct and collect a small amount of long CoT data to fine-tune the model as the initial RL actor. To collect such data, we have explored several approaches: using few-shot prompting with a long CoT as an example, directly prompting models to generate detailed answers with reflection and verification, gathering DeepSeek-R1Zero outputs in a readable format, and refining the results through post-processing by human annotators.

In this work, we collect thousands of cold-start data to fine-tune the DeepSeek-V3-Base as the starting point for RL. Compared to DeepSeek-R1-Zero, the advantages of cold start data

"""

    # test_content = r"""$\left\{o_{1}, o_{2}, \cdots, o_{G}\right\}$"""

    test_content = r"""\subsection*{2.3.1. Cold Start}"""

    test_content = r"""
Question: If $a>1$, then the sum of the real solutions of $\sqrt{a-\sqrt{a+x}}=x$ is equal to

Response: <think>

To solve the equation $\sqrt{a-\sqrt{a+x}}=x$, let's start by squaring both $\cdots$

$(\sqrt{a-\sqrt{a+x}})^{2}=x^{2} \Longrightarrow a-\sqrt{a+x}=x^{2}$.

Rearrange to isolate the inner square root term:

$\left(a-x^{2}\right)^{2}=a+x \Longrightarrow a^{2}-2 a x^{2}+\left(x^{2}\right)^{2}=a+x \Longrightarrow x^{4}-2 a x^{2}-x+\left(a^{2}-a\right)=0$

Wait, wait. Wait. That's an aha moment I can flag here.

Let's reevaluate this step-by-step to identify if the correct sum can be $\cdots$

We started with the equation:

$\sqrt{a-\sqrt{a+x}}=x$

First, let's square both sides:

$a-\sqrt{a+x}=x^{2} \Longrightarrow \sqrt{a+x}=a-x^{2}$

Next, I could square both sides again, treating the equation: $\cdots$

Table 3 | An interesting "aha moment" of an intermediate version of DeepSeek-R1-Zero. The model learns to rethink using an anthropomorphic tone. This is also an aha moment for us, allowing us to witness the power and beauty of reinforcement learning.

Drawback of DeepSeek-R1-Zero Although DeepSeek-R1-Zero exhibits strong reasoning capabilities and autonomously develops unexpected and powerful reasoning behaviors, it faces several issues. For instance, DeepSeek-R1-Zero struggles with challenges like poor readability, and language mixing. To make reasoning processes more readable and share them with the open community, we explore DeepSeek-R1, a method that utilizes RL with human-friendly cold-start data.

\subsection*{2.3. DeepSeek-R1: Reinforcement Learning with Cold Start}

Inspired by the promising results of DeepSeek-R1-Zero, two natural questions arise: 1) Can reasoning performance be further improved or convergence accelerated by incorporating a small amount of high-quality data as a cold start? 2) How can we train a user-friendly model that not only produces clear and coherent Chains of Thought (CoT) but also demonstrates strong general capabilities? To address these questions, we design a pipeline to train DeepSeek-R1. The pipeline consists of four stages, outlined as follows.

\subsection*{2.3.1. Cold Start}

Unlike DeepSeek-R1-Zero, to prevent the early unstable cold start phase of RL training from the base model, for DeepSeek-R1 we construct and collect a small amount of long CoT data to fine-tune the model as the initial RL actor. To collect such data, we have explored several approaches: using few-shot prompting with a long CoT as an example, directly prompting models to generate detailed answers with reflection and verification, gathering DeepSeek-R1Zero outputs in a readable format, and refining the results through post-processing by human annotators.

In this work, we collect thousands of cold-start data to fine-tune the DeepSeek-V3-Base as the starting point for RL. Compared to DeepSeek-R1-Zero, the advantages of cold start data
    
    """

    optimized = optimizer.optimize(test_content)
    print("优化前:")
    print(test_content)
    print("\n优化后:")
    print(optimized)