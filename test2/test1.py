from ant4latex import Lexer, Parser, Token

# 基本使用
latex_code = r"""
\documentclass{article}
\begin{document}
Hello \textbf{World}!
\end{document}
"""

lexer = Lexer()
tokens = lexer.lex(latex_code)
parser = Parser()
ast = parser.parse(tokens)