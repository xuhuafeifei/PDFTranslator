"""
Persistence Layer - 持久化层
支持LaTeX和PDF输出策略
"""
import os
import re
import cv2
import math
from abc import ABC, abstractmethod
from typing import List, Tuple


class PersistenceStrategy(ABC):
    """持久化策略基类"""
    
    @abstractmethod
    def save(self, content: List[str], output_path: str, **kwargs):
        """
        保存内容
        
        Args:
            content: 内容列表
            output_path: 输出路径
            **kwargs: 其他参数
        """
        pass


class LatexPersistenceStrategy(PersistenceStrategy):
    """LaTeX持久化策略"""
    
    def __init__(self):
        self.latex_template = """% !TEX root = {filename}
% !TEX program = xelatex
\\documentclass[12pt,a4paper]{{report}} % 也可以改成 report 或 book

% ----------------------
% 中文支持
% ----------------------
\\usepackage[UTF8]{{ctex}}          % 中文支持
\\setCJKmainfont{{Songti SC}}
\\setCJKsansfont{{Heiti SC}}
\\setCJKmonofont{{STFangsong}}
\\setmainfont{{Times New Roman}}     % 英文字体
\\setsansfont{{Arial}}               % 英文无衬线
\\setmonofont{{Courier New}}         % 英文等宽

% ----------------------
% 数学和符号
% ----------------------
\\usepackage{{amsmath, amssymb, amsfonts}} 
\\usepackage{{geometry}}            % 页面边距
\\geometry{{left=3cm, right=3cm, top=2.5cm, bottom=2.5cm}}

% ----------------------
% 其他常用宏包
% ----------------------
\\usepackage{{graphicx}}            % 图片
\\usepackage{{hyperref}}            % 超链接
\\usepackage{{caption}}             % 图表标题

% ----------------------
% 文档开始
% ----------------------
\\begin{{document}}

{content}

% ----------------------
% 文档结束
% ----------------------
\\end{{document}}"""
    
    def save(self, content: List[str], output_path: str, **kwargs):
        """保存为LaTeX文件"""
        # 合并内容
        combined_content = "\n".join(content)
        
        # 添加maketitle
        combined_content = combined_content.replace("\\begin{abstract}", "\\maketitle\n\\begin{abstract}")
        
        # 获取文件名（不含扩展名）
        filename = os.path.basename(output_path)
        if '.' in filename:
            filename = filename.rsplit('.', 1)[0]
        
        # 生成完整的LaTeX文档
        latex_content = self.latex_template.format(filename=filename, content=combined_content)
        
        # 保存文件
        tex_path = output_path + ".case_tag.tex"
        with open(tex_path, 'w', encoding='utf-8') as f:
            f.write(latex_content)
        
        print(f"LaTeX文件已保存: {tex_path}")


class PdfPersistenceStrategy(PersistenceStrategy):
    """PDF持久化策略（空壳实现）"""
    
    def save(self, content: List[str], output_path: str, **kwargs):
        """
        保存为PDF文件
        
        注意：这是一个空壳实现，需要你自己完成
        """
        print("PDF持久化策略 - 空壳实现")
        print(f"内容长度: {len(content)}")
        print(f"输出路径: {output_path}")
        print("请实现PDF生成逻辑...")
        
        # TODO: 实现PDF生成逻辑
        # 这里可以集成LaTeX编译、或者其他PDF生成库
        pass


class ImageProcessor:
    """图片处理器"""
    
    @staticmethod
    def smart_resize(height: int, width: int, factor: int = 28, 
                     min_pixels: int = 3136, max_pixels: int = 1024*1024) -> Tuple[int, int]:
        """
        智能调整图片尺寸
        
        Args:
            height: 原始高度
            width: 原始宽度
            factor: 因子
            min_pixels: 最小像素数
            max_pixels: 最大像素数
            
        Returns:
            调整后的(高度, 宽度)
        """
        if height < factor or width < factor:
            raise ValueError(f"height:{height} or width:{width} must be larger than factor:{factor}")
        elif max(height, width) / min(height, width) > 200:
            raise ValueError(
                f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
            )
        
        h_bar = round(height / factor) * factor
        w_bar = round(width / factor) * factor
        
        if h_bar * w_bar > max_pixels:
            beta = math.sqrt((height * width) / max_pixels)
            h_bar = math.floor(height / beta / factor) * factor
            w_bar = math.floor(width / beta / factor) * factor
        elif h_bar * w_bar < min_pixels:
            beta = math.sqrt(min_pixels / (height * width))
            h_bar = math.ceil(height * beta / factor) * factor
            w_bar = math.ceil(width * beta / factor) * factor
        
        return h_bar, w_bar
    
    @staticmethod
    def get_fig_name_by_div(html_content: str, img_div: str) -> str:
        """通过img_div确定html_content中图片名称"""
        try:
            html_content_lines = [line for line in html_content.split("\n") if line.strip() != ""]
            for i, line in enumerate(html_content_lines):
                if img_div in line:
                    return re.sub(
                        r'<p\b[^>]*>(.*?)<\/p>',
                        r'\1\n',
                        html_content_lines[i+1],
                        flags=re.DOTALL | re.IGNORECASE,
                    ).strip()
        except Exception as e:
            print(f"获取图片名称失败: {e}")
            return ""  # 返回空字符串而不是错误信息
        return ""  # 返回空字符串而不是错误信息
    
    @staticmethod
    def replace_img_with_includegraphics_and_save_img(img, scale: Tuple[float, float], 
                                                   html_content: str, page_num: int, 
                                                   output_dir: str, 
                                                   image_name_template: str = "page_{page_num}_{number}.png") -> str:
        """
        将img标签替换为includegraphics标签并保存图片
        
        Args:
            img: 图片数组
            scale: 缩放比例
            html_content: HTML内容
            page_num: 页码
            output_dir: 输出目录
            image_name_template: 图片名称模板
            
        Returns:
            处理后的HTML内容
        """
        # 匹配自闭合的img标签
        pattern = re.compile(r'<img\s+data-bbox="(\d+),(\d+),(\d+),(\d+)"\s*/>')
        
        # 找到所有匹配
        matches = pattern.findall(html_content)
        print(f"找到 {len(matches)} 个img标签")
        
        # 逐个处理每个匹配
        num = 1
        result = html_content
        for i, match in enumerate(matches):
            x1, y1, x2, y2 = match
            print(f"处理第 {i+1} 个img标签: 坐标({x1},{y1},{x2},{y2})")
            
            # 转换坐标为整数
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            
            # 计算原始图片尺寸
            width = x2 - x1
            height = y2 - y1
            
            # 解包缩放比例
            scale_x, scale_y = scale
            
            # 计算缩放后的坐标（用于截取图片）
            scaled_x1 = int(x1 * scale_x)
            scaled_y1 = int(y1 * scale_y)
            scaled_x2 = int(x2 * scale_x)
            scaled_y2 = int(y2 * scale_y)
            
            print(f"  原始尺寸: {width} x {height}")

            MAX_WIDTH = 500
            # 尺寸缩放, 如果宽度大于580, 等比例缩小
            if width > MAX_WIDTH:
                alpha = MAX_WIDTH / width
                width = MAX_WIDTH
                height = int(height * alpha)
            
            # 生成图片路径
            image_name = image_name_template.format(page_num=page_num, number=num)
            print(f"  页码: {page_num}, 序号: {num}, 图片路径: {image_name}")
            
            # 获取旧标签
            old_tag = f'<img data-bbox="{x1},{y1},{x2},{y2}"/>'
            # 获取图片名称
            fig_name = ImageProcessor.get_fig_name_by_div(html_content, old_tag)
            # 构建新的标签（包含尺寸和居中）
            if fig_name.strip():  # 如果图片名称不为空
                new_tag = f'\\begin{{center}}\n\\includegraphics[width={width}pt, height={height}pt]{{{image_name}}}\n\\end{{center}}\n\\begin{{center}}\n{fig_name}\n\\end{{center}}\n'
            else:  # 如果图片名称为空，只显示图片
                new_tag = f'\\begin{{center}}\n\\includegraphics[width={width}pt, height={height}pt]{{{image_name}}}\n\\end{{center}}\n'
            
            # 替换第一个匹配（避免重复替换）
            if fig_name.strip():  # 只有当图片名称不为空时才替换
                result = result.replace(fig_name, "", 1)
            result = result.replace(old_tag, new_tag, 1)
            print(f"  替换: {old_tag} -> {new_tag}")
            
            print(f"截取并保存图片...")
            # 截取并保存图片（使用缩放后的坐标）
            cropped_img = img[scaled_y1:scaled_y2, scaled_x1:scaled_x2]
            # 保存图片
            image_path = os.path.join(output_dir, image_name)
            cv2.imwrite(image_path, cropped_img)
            print(f"截取并保存图片完成...")
            num += 1
        
        return result
    
    @staticmethod
    def process_images_and_replace_predictions(image_paths: List[str], html_contents: List[str], 
                                             output_path: str, input_heights: List[int], 
                                             input_widths: List[int]) -> List[str]:
        """
        处理图片并替换预测结果中的img标签
        
        Args:
            image_paths: 图片路径列表
            html_contents: HTML内容列表
            output_path: 输出路径
            input_heights: 输入高度列表
            input_widths: 输入宽度列表
            
        Returns:
            处理后的HTML内容列表
        """
        processed_contents = []
        output_dir = os.path.dirname(output_path)
        
        for i, (image_path, html_content, input_height, input_width) in enumerate(
            zip(image_paths, html_contents, input_heights, input_widths)):
            
            img = cv2.imread(image_path)
            img_height, img_width, _ = img.shape
            scale = (img_width / input_width, img_height / input_height)
            
            processed_content = ImageProcessor.replace_img_with_includegraphics_and_save_img(
                img, scale, html_content, i + 1, output_dir
            )
            processed_contents.append(processed_content)
        
        return processed_contents
    

    @staticmethod
    def process_table_and_replace_predictions(image_paths: List[str], html_contents: List[str], 
                                             output_path: str, input_heights: List[int], 
                                             input_widths: List[int]) -> List[str]:
        """
        处理图片并替换预测结果中的table标签
        
        Args:
            image_paths: 图片路径列表
            html_contents: HTML内容列表
            output_path: 输出路径
            input_heights: 输入高度列表
            input_widths: 输入宽度列表
            
        Returns:
            处理后的HTML内容列表
        """
        processed_contents = []
        output_dir = os.path.dirname(output_path)

        
        for i, (image_path, html_content, input_height, input_width) in enumerate(
            zip(image_paths, html_contents, input_heights, input_widths)):
            
            img = cv2.imread(image_path)
            img_height, img_width, _ = img.shape
            scale = (img_width / input_width, img_height / input_height)

            # 处理html, 将table标签转化为img标签
            lines = html_content.split("\n")
            new_lines = []
            for line in lines:
                if "class=\"table\"" in line:
                    pattern = re.compile(r'data-bbox="(\d+),(\d+),(\d+),(\d+)">')
                    # 找到所有匹配
                    matches = pattern.findall(line)
                    for match in matches:
                        x1, y1, x2, y2 = match
                        # 将table标签替换为img标签
                        new_lines.append(f"<img data-bbox='{x1},{y1},{x2},{y2}'/>")
                else:
                    new_lines.append(line)
            html_content = "\n".join(new_lines)
            
            processed_content = ImageProcessor.replace_img_with_includegraphics_and_save_img(
                img, scale, html_content, i + 1, output_dir
            )
            processed_contents.append(processed_content)
        
        return processed_contents

class PersistenceLayer:
    """持久化层"""
    
    def __init__(self, strategy: PersistenceStrategy = None):
        self.strategy = strategy or LatexPersistenceStrategy()
    
    def set_strategy(self, strategy: PersistenceStrategy):
        """设置持久化策略"""
        self.strategy = strategy
    
    def save(self, content: List[str], output_path: str, **kwargs):
        """保存内容"""
        self.strategy.save(content, output_path, **kwargs)
