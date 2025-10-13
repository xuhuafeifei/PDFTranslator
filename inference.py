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

def plot_bbox(img_path, pred, input_height, input_width, output_path):
    img = cv2.imread(img_path)
    img_height, img_width, _ = img.shape
    scale = (img_width / input_width, img_height / input_height)
    bboxes = []

    pattern = re.compile(r'data-bbox="(\d+),(\d+),(\d+),(\d+)"')

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
    prediction = qwenvl_pred_cast_tag(prediction)
    with open(output_path + ".case_tag", 'w') as f:
        f.write(prediction)

def write_plot_bbox(image_path, prediction, h, w):
    output_img_path = image_path.split(".")[0]+"_vis.png"
    plot_bbox(image_path, prediction, h, w, output_img_path)

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

    args = parser.parse_args()
    

    model_path = args.model_path
    prompt = args.prompt
    output_path =  args.output_path
    attn = args.attn
    device = args.device
    print(device)
    device_map = "cuda" if device == "cuda" else "cpu"
    print(attn)
    print(device)
    print(device_map)
    print(device_map)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16, attn_implementation=attn, device_map=device_map)
    print("model loaded")
    processor = AutoProcessor.from_pretrained(model_path)
    print("processor loaded")

    if args.image_path:
        image_path = args.image_path
        prediction, h, w= inference(image_path, prompt)
        print(f"写入检测区域图片...")
        write_plot_bbox(image_path, prediction, h, w)
        print(f"持久化预测结果...")
        write_prediction(prediction, output_path)
    else:
        pdf_path = args.pdf_path
        converter = PDFToImageConverter()
        convert_temp_path = converter.convert_pdf_to_images_temp(pdf_path)
        try:
            prediction_list = []
            # for循环处理所有内容
            for i, temp_path in enumerate(convert_temp_path):
                print(f"处理第{i+1}页...")
                prediction, h, w= inference(temp_path, prompt)
                print(f"处理第{i+1}页完成...")
                prediction_list.append(prediction)
                print(f"写入检测区域图片...")
                write_plot_bbox(temp_path, prediction, h, w)
            print(f"持久化预测结果...")
            write_prediction("\n".join(prediction_list), output_path)
        finally:
            converter.cleanup_temp_images(convert_temp_path)



