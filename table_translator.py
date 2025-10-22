from bs4 import BeautifulSoup
import re

# 废弃，使用html_table_to_latex_simple
def html_table_to_latex(html_table: str) -> str:
    """
    将HTML表格转换为LaTeX表格
    
    Args:
        html_table: HTML表格字符串
        
    Returns:
        LaTeX表格字符串
    """
    soup = BeautifulSoup(html_table, "html.parser")
    table = soup.find("table")
    
    if not table:
        return ""

    # 提取所有行
    rows = table.find_all("tr")
    if not rows:
        return ""

    # 计算实际列数（考虑colspan）
    def count_effective_cols(row):
        cols = row.find_all(["td", "th"])
        total_cols = 0
        for col in cols:
            colspan = int(col.get("colspan", 1))
            total_cols += colspan
        return total_cols
    
    # 找到最大有效列数
    num_cols = max(count_effective_cols(row) for row in rows)
    
    latex_lines = []
    
    # 添加表格开始
    latex_lines.append("\\begin{longtable}{|" + "c|" * num_cols + "}")
    latex_lines.append("\\hline")

    # 处理每一行
    for i, row in enumerate(rows):
        cols = row.find_all(["td", "th"])
        cell_texts = [""] * num_cols  # 初始化所有列为空
        col_index = 0  # 当前列索引

        for c in cols:
            text = c.get_text(strip=True)
            # 不进行特殊字符转义，保持原样
            
            # 处理colspan和rowspan
            colspan = int(c.get("colspan", 1))
            rowspan = int(c.get("rowspan", 1))

            # 找到下一个可用的列位置
            while col_index < num_cols and cell_texts[col_index] != "":
                col_index += 1
            
            if col_index >= num_cols:
                break

            if colspan > 1 and rowspan > 1:
                # 同时有colspan和rowspan
                cell_texts[col_index] = f"\\multirow{{{rowspan}}}{{*}}{{\\multicolumn{{{colspan}}}{{|c|}}{{{text}}}}}"
            elif colspan > 1:
                # 只有colspan
                cell_texts[col_index] = f"\\multicolumn{{{colspan}}}{{|c|}}{{{text}}}"
            elif rowspan > 1:
                # 只有rowspan
                cell_texts[col_index] = f"\\multirow{{{rowspan}}}{{*}}{{{text}}}"
            else:
                # 普通单元格
                cell_texts[col_index] = text

            # 标记被colspan占用的列
            for j in range(colspan):
                if col_index + j < num_cols:
                    if j == 0:
                        continue  # 第一列已经设置了内容
                    else:
                        cell_texts[col_index + j] = ""  # 其他列标记为空，但会被multicolumn处理
            
            col_index += colspan

        line = " & ".join(cell_texts) + " \\\\"
        latex_lines.append(line)
        
        # 添加分隔线
        if i == 0:  # 表头后
            latex_lines.append("\\hline")
        elif i < len(rows) - 1:  # 中间行
            latex_lines.append("\\hline")

    # 添加表格结束
    latex_lines.append("\\hline")
    latex_lines.append("\\end{longtable}")

    return "\n".join(latex_lines)


def escape_latex_special_chars(text: str) -> str:
    """
    转义LaTeX特殊字符
    
    Args:
        text: 原始文本
        
    Returns:
        转义后的文本
    """
    # LaTeX特殊字符映射
    special_chars = {
        '&': '\\&',
        '%': '\\%',
        '$': '\\$',
        '#': '\\#',
        '^': '\\textasciicircum{}',
        '_': '\\_',
        '{': '\\{',
        '}': '\\}',
        '~': '\\textasciitilde{}',
        '\\': '\\textbackslash{}',
    }
    
    # 处理 | 符号（不在数学模式中）
    text = re.sub(r'(?<!\$)\|(?!\$)', r'\\textbar{}', text)
    
    # 替换其他特殊字符
    for char, replacement in special_chars.items():
        text = text.replace(char, replacement)
    
    return text


def html_table_to_latex_simple(html_table: str) -> str:
    """
    简化版HTML表格转LaTeX（不使用longtable）
    
    Args:
        html_table: HTML表格字符串
        
    Returns:
        LaTeX表格字符串
    """
    soup = BeautifulSoup(html_table, "html.parser")
    table = soup.find("table")
    
    if not table:
        return ""

    rows = table.find_all("tr")
    if not rows:
        return ""

    # 计算实际列数（考虑colspan）
    def count_effective_cols(row):
        cols = row.find_all(["td", "th"])
        total_cols = 0
        for col in cols:
            colspan = int(col.get("colspan", 1))
            total_cols += colspan
        return total_cols
    
    num_cols = max(count_effective_cols(row) for row in rows)
    
    latex_lines = []
    
    # 添加表格开始
    latex_lines.append("\\begin{table}[h]")
    latex_lines.append("\\centering")
    latex_lines.append("\\begin{tabular}{|" + "c|" * num_cols + "}")
    latex_lines.append("\\hline")

    # 处理每一行
    for i, row in enumerate(rows):
        cols = row.find_all(["td", "th"])
        cell_texts = []

        for c in cols:
            text = c.get_text(strip=True)
            # text = escape_latex_special_chars(text)
            
            colspan = int(c.get("colspan", 1))
            if colspan > 1:
                cell_texts.append(f"\\multicolumn{{{colspan}}}{{|c|}}{{{text}}}")
            else:
                cell_texts.append(text)

        # 确保每行都有足够的列
        while len(cell_texts) < num_cols:
            cell_texts.append("")

        line = " & ".join(cell_texts) + " \\\\"
        latex_lines.append(line)
        
        if i == 0:
            latex_lines.append("\\hline")

    # 添加表格结束
    latex_lines.append("\\hline")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\end{table}")

    return "\n".join(latex_lines)


if __name__ == "__main__":
    # 示例：你可以把整张表直接复制到这里
    html_input = """
    <table><tr><td>Benchmark (Metric)</td><td>Claude-3.5-Sonnet-1022</td><td>GPT-4o 0513</td><td>DeepSeek V3</td><td>OpenAI o1-mini</td><td>OpenAI o1-1217</td><td>DeepSeek R1</td></tr><tr><td>Architecture</td><td>-</td><td>-</td><td>MoE</td><td>-</td><td>-</td><td>MoE</td></tr><tr><td>\# Activated Params</td><td>-</td><td>-</td><td>37B</td><td>-</td><td>-</td><td>37B</td></tr><tr><td>\# Total Params</td><td>-</td><td>-</td><td>671B</td><td>-</td><td>-</td><td>671B</td></tr><tr><td rowspan="9">English</td><td>MMLU (Pass@1)</td><td>88.3</td><td>87.2</td><td>88.5</td><td>85.2</td><td>91.8</td><td>90.8</td></tr><tr><td>MMLU-Redux (EM)</td><td>88.9</td><td>88.0</td><td>89.1</td><td>86.7</td><td>-</td><td>92.9</td></tr><tr><td>MMLU-Pro (EM)</td><td>78.0</td><td>72.6</td><td>75.9</td><td>80.3</td><td>-</td><td>84.0</td></tr><tr><td>DROP (3-shot FI)</td><td>88.3</td><td>83.7</td><td>91.6</td><td>83.9</td><td>90.2</td><td>92.2</td></tr><tr><td>IF-Eval (Prompt Strict)</td><td>86.5</td><td>84.3</td><td>86.1</td><td>84.8</td><td>-</td><td>83.3</td></tr><tr><td>GPQA Diamond (Pass@1)</td><td>65.0</td><td>49.9</td><td>59.1</td><td>60.0</td><td>75.7</td><td>71.5</td></tr><tr><td>SimpleQA (Correct)</td><td>28.4</td><td>38.2</td><td>24.9</td><td>7.0</td><td>47.0</td><td>30.1</td></tr><tr><td>FRAMES (Acc.)</td><td>72.5</td><td>80.5</td><td>73.3</td><td>76.9</td><td>-</td><td>82.5</td></tr><tr><td>AlpacaEval2.0 (LC-winrate)</td><td>52.0</td><td>51.1</td><td>70.0</td><td>57.8</td><td>-</td><td>87.6</td></tr><tr><td>ArenaHard (GPT-4-1106)</td><td>85.2</td><td>80.4</td><td>85.5</td><td>92.0</td><td>-</td><td>92.3</td></tr><tr><td rowspan="5">Code</td><td>LiveCodeBench (Pass@1-COT)</td><td>38.9</td><td>32.9</td><td>36.2</td><td>53.8</td><td>63.4</td><td>65.9</td></tr><tr><td>Codeforces (Percentile)</td><td>20.3</td><td>23.6</td><td>58.7</td><td>93.4</td><td>96.6</td><td>96.3</td></tr><tr><td>Codeforces (Rating)</td><td>717</td><td>759</td><td>1134</td><td>1820</td><td>2061</td><td>2029</td></tr><tr><td>SWE Verified (Resolved)</td><td>50.8</td><td>38.8</td><td>42.0</td><td>41.6</td><td>48.9</td><td>49.2</td></tr><tr><td>Aider-Polyglot (Acc.)</td><td>45.3</td><td>16.0</td><td>49.6</td><td>32.9</td><td>61.7</td><td>53.3</td></tr><tr><td rowspan="3">Math</td><td>AIME 2024 (Pass@1)</td><td>16.0</td><td>9.3</td><td>39.2</td><td>63.6</td><td>79.2</td><td>79.8</td></tr><tr><td>MATH-500 (Pass@1)</td><td>78.3</td><td>74.6</td><td>90.2</td><td>90.0</td><td>96.4</td><td>97.3</td></tr><tr><td>CNMO 2024 (Pass@1)</td><td>13.1</td><td>10.8</td><td>43.2</td><td>67.6</td><td>-</td><td>78.8</td></tr><tr><td rowspan="3">Chinese</td><td>CLUEWSC (EM)</td><td>85.4</td><td>87.9</td><td>90.9</td><td>89.9</td><td>-</td><td>92.8</td></tr><tr><td>C-Eval (EM)</td><td>76.7</td><td>76.0</td><td>86.5</td><td>68.9</td><td>-</td><td>91.8</td></tr><tr><td>C-SimpleQA (Correct)</td><td>55.4</td><td>58.7</td><td>68.0</td><td>40.3</td><td>-</td><td>63.7</td></tr></table> 
    """

    latex_output = html_table_to_latex(html_input)
    print("start----------")
    print(latex_output)
