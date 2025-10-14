import torch
import os 
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from PIL import Image,ImageFont,ImageDraw
import os 
import re
import math 
import cv2 
import argparse
from pdf_to_image import convert_pdf_to_images_temp, PDFToImageConverter
from translator import TextTranslator


def inference(img_url, prompt, system_prompt="You are a helpful assistant"):
    print("开始推理...")
    image = Image.open(img_url)
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
            "image": img_url
            }
        ]
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt").to(device_map)

    print("准备生成...")
    output_ids = model.generate(**inputs, max_new_tokens=8192)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    input_height = inputs['image_grid_thw'][0][1]*14
    input_width = inputs['image_grid_thw'][0][2]*14

    print("生成完成...")
    return output_text[0], input_height, input_width



def qwenvl_pred_cast_tag(input_text: str) -> str:
    output = input_text

    IMG_RE = re.compile(
        r'<img\b[^>]*\bdata-bbox\s*=\s*"?\d+,\d+,\d+,\d+"?[^>]*\/?>',
        flags=re.IGNORECASE,
    )
    output = IMG_RE.sub('', output)


    output = re.sub(
        r'<p\b[^>]*>(.*?)<\/p>',
        r'\1\n',
        output,
        flags=re.DOTALL | re.IGNORECASE,
    )


    def strip_div(class_name: str, txt: str) -> str:
        pattern = re.compile(
            rf'\s*<div\b[^>]*class="{class_name}"[^>]*>(.*?)<\/div>\s*',
            flags=re.DOTALL | re.IGNORECASE,
        )
        if class_name == 'image':
            # 图片不能和上下文本贴合
            return pattern.sub(r' \n\n\1\n\n ', txt)
        return pattern.sub(r' \1 ', txt)

    for cls in ['image', 'chemistry', 'table', 'formula', 'image caption']:
        output = strip_div(cls, output)
    output = output.replace(" </td>", "</td>")
    return output


def smart_resize(
    height: int, width: int, factor: int = 28, min_pixels: int = 3136, max_pixels: int = 1024*1024
):
    """Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.

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

'''
通过img_div确定html_content中图片名称
'''
def get_fig_name_by_div(html_content, img_div) -> str:
    # html按照换行符分割
    try:
        # 切除换行的同时, 去除所有的空行
        html_content_lines = [line for line in html_content.split("\n") if line.strip() != ""]
        for i, line in enumerate(html_content_lines):
            if img_div in line:
                # 返回下一行内容, 并去除下一行的所有标签内容
                return re.sub(
                    r'<p\b[^>]*>(.*?)<\/p>',
                    r'\1\n',
                    html_content_lines[i+1],
                    flags=re.DOTALL | re.IGNORECASE,
                ).strip()
    except Exception as e:
        print(f"获取图片名称失败: {e}")
        return "get_fig_name_error"
    return "can_not_find_fig_name"

'''
获取图片, 并替换预测结果中的img标签为includegraphics标签, 并保存图片
img: 图片
scale: 缩放
html_content: HTML内容
page_num: 页码
output_dir: 输出目录
image_name_template: 图片名称模板
'''
def replace_img_with_includegraphics_and_save_img(img, scale, html_content, page_num, output_dir, image_name_template="page_{page_num}_{number}.png"):
    """
    将自闭合的img标签替换为包含includegraphics的完整标签
    
    参数:
        html_content: HTML内容
        image_path_template: 图片路径模板，{page_num}会被替换为页码
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
        
        # 生成图片路径
        image_name = image_name_template.format(page_num=page_num, number=num)
        print(f"  页码: {page_num}, 序号: {num}, 图片路径: {image_name}")

        # 获取旧标签
        old_tag = f'<img data-bbox="{x1},{y1},{x2},{y2}"/>'
        # 获取图片名称
        fig_name = get_fig_name_by_div(html_content, old_tag)
        # 构建新的标签（包含尺寸和居中）
        new_tag = f'\\begin{{center}}\n\\includegraphics[width={width}pt, height={height}pt]{{{image_name}}}\n\\end{{center}}\n\\begin{{center}}\n{fig_name}\n\\end{{center}}'

        # 替换第一个匹配（避免重复替换）
        result = result.replace(fig_name, "", 1).replace(old_tag, new_tag, 1)
        print(f"  替换: {old_tag} -> {new_tag}")

        print(f"截取并保存图片...")
        # 截取并保存图片（使用缩放后的坐标）
        # 截取img的矩形区域
        cropped_img = img[scaled_y1:scaled_y2, scaled_x1:scaled_x2]
        # 保存图片
        image_path = os.path.join(output_dir, image_name)
        cv2.imwrite(image_path, cropped_img)
        print(f"截取并保存图片完成...")
        num += 1

    return result


'''
获取图片, 并替换预测结果中的img标签为includegraphics标签, 并保存图片
img_path: 图片路径
pred: 预测结果
output_path: 输出路径
input_height: 输入高度
input_width: 输入宽度
page_num: 页码
'''
def catch_picture_and_replace_prediction(img_path, pred, output_path, input_height, input_width, page_num) -> str: 
    img = cv2.imread(img_path)
    img_height, img_width, _ = img.shape
    scale = (img_width / input_width, img_height / input_height)

    # 取出目录部分，去除output_path最后的文件名
    output_dir = os.path.dirname(output_path)
    return replace_img_with_includegraphics_and_save_img(img, scale, pred, page_num, output_dir)


def plot_bbox(img_path, pred, input_height, input_width, output_path):
    img = cv2.imread(img_path)
    img_height, img_width, _ = img.shape
    scale = (img_width / input_width, img_height / input_height)
    bboxes = []

    pattern = re.compile(r'img data-bbox="(\d+),(\d+),(\d+),(\d+)"')

    scale_x, scale_y = scale  

    def replace_bbox(match):
        x1, y1, x2, y2 = map(int, match)
        bboxes.append([int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y)])

    
    matches = re.findall(pattern, pred)
    if matches:
        for match in matches:
            # print(match)
            replace_bbox(match)
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 8)

    cv2.imwrite(output_path, img)

def write_prediction(prediction: str, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(prediction)
    
    # 处理预测结果
    prediction = qwenvl_pred_cast_tag(prediction)
    
    # 添加LaTeX模板
    latex_template = """% !TEX root = {filename}
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
    
    # 获取文件名（不含扩展名）
    filename = os.path.basename(output_path)
    if '.' in filename:
        filename = filename.rsplit('.', 1)[0]
    
    # maketitle
    predication = prediction.replace(f"\\begin{{abstract}}", "\\maketitle\n\\begin{{abstract}}")
    # 生成完整的LaTeX文档
    latex_content = latex_template.format(filename=filename, content=prediction)
    
    with open(output_path + ".case_tag.tex", 'w', encoding='utf-8') as f:
        f.write(latex_content)

def write_plot_bbox(image_path, prediction, h, w):
    output_img_path = image_path.split(".")[0]+"_vis.png"
    plot_bbox(image_path, prediction, h, w, output_img_path)

def translate_prediction(prediction: str) -> str:
    result = translator.translate(prediction)
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Logics-Parsing for document parsing and visualize the output.")

    parser.add_argument("--model_path", type=str, required=True, 
                        help="Path to the directory containing the pre-trained model and processor.")
    # 创建互斥参数组，必须选择其中一个
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--image_path", type=str,
                            help="Path to the input image file for parsing.")
    input_group.add_argument("--pdf_path", type=str,
                            help="Path to the input pdf file for parsing.")
    parser.add_argument("--output_path", type=str, required=True, 
                        help="Path to save the prediction.")
    parser.add_argument("--prompt", type=str, default="QwenVL HTML", 
                        help="The prompt to send to the model. (default: %(default)s)")
    parser.add_argument("--attn", type=str, default="sdpa",
                        help="The attention implementation to use. (default: %(default)s)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="The device to use. (default: %(default)s)")
    parser.add_argument("--translator_model_path", type=str, default=None,
                        help="Path to the translator model (optional)")

    args = parser.parse_args()
    

    model_path = args.model_path
    prompt = args.prompt
    output_path =  args.output_path
    attn = args.attn
    device = args.device
    device_map = "cuda" if device == "cuda" else "cpu"
    print(attn)
    print(device)
    print(device_map)
    print(device_map)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16, attn_implementation=attn, device_map=device_map)
    translator = TextTranslator(model_path=args.translator_model_path)
    print("model loaded")
    processor = AutoProcessor.from_pretrained(model_path)
    print("processor loaded")

    if args.image_path:
        # 图片推理
        image_path = args.image_path
        prediction, h, w = inference(image_path, prompt)
        prediction = catch_picture_and_replace_prediction(image_path, prediction, output_path, h, w, 1)

        # 翻译预测结果
        try:
            print(f"翻译预测结果...")
            translator.load_model()
            prediction = translate_prediction(prediction)
        except Exception as e:
            print(f"翻译预测结果失败: {e}")
        finally:
            # 卸载翻译模型
            translator.unload_model()

        print(f"持久化预测结果...")
        write_prediction(prediction, output_path)
    else:
        pdf_path = args.pdf_path
        converter = PDFToImageConverter()
        convert_temp_path = converter.convert_pdf_to_images_temp(pdf_path)
        try:
            prediction_list = []
            # for循环pdf解析
            for i, temp_path in enumerate(convert_temp_path):
                print(f"处理第{i+1}页...")
                prediction, h, w= inference(temp_path, prompt)
                prediction = catch_picture_and_replace_prediction(temp_path, prediction, output_path, h, w, i+1)
                print(f"处理第{i+1}页完成...")
                prediction_list.append(prediction)
            # 翻译预测结果
            try:
                translator.load_model()
                print(f"翻译预测结果...")
                prediction_list = [translate_prediction(prediction) for prediction in prediction_list]
            except Exception as e:
                print(f"翻译预测结果失败: {e}")
            finally:
                # 卸载翻译模型
                translator.unload_model()
            print(f"持久化预测结果...")
            write_prediction("\n".join(prediction_list), output_path)
        finally:
            converter.cleanup_temp_images(convert_temp_path)



