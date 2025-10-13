"""
PDF转图片转换器模块

本模块提供将PDF文件转换为图片的功能，支持两种模式：
1. 临时模式：创建临时目录，转换PDF，返回图片路径，然后清理
2. 永久模式：将PDF转换为图片并保存到指定的输出目录

依赖包：
- pdf2image: 用于PDF转图片转换
- Pillow: 用于图像处理
- pathlib: 用于路径处理
- tempfile: 用于临时目录管理
"""

import os
import tempfile
import shutil
import argparse
import sys
from pathlib import Path
from typing import List, Optional, Union
import logging

try:
    from pdf2image import convert_from_path
    from PIL import Image
except ImportError as e:
    raise ImportError(
        "缺少必需的包。请使用以下命令安装："
        "pip install pdf2image pillow"
    ) from e

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PDFToImageConverter:
    """
    处理PDF转图片转换的类，支持临时和永久两种模式。
    """

    def __init__(self, dpi: int = 300, output_format: str = 'PNG'):
        """
        初始化PDF转图片转换器。

        参数:
            dpi (int): 图片转换的DPI（默认: 300）
            output_format (str): 输出图片格式（默认: 'PNG'）
        """
        self.dpi = dpi
        self.output_format = output_format.upper()

    def convert_pdf_to_images_temp(self, pdf_path: Union[str, Path]) -> List[str]:
        """
        使用临时目录将PDF转换为图片。
        创建临时目录，转换PDF，返回图片路径，然后清理。

        参数:
            pdf_path (Union[str, Path]): 输入PDF文件的路径

        返回:
            List[str]: 临时图片文件路径列表

        异常:
            FileNotFoundError: 如果PDF文件不存在
            ValueError: 如果PDF路径无效
            Exception: 如果转换失败
        """
        pdf_path = Path(pdf_path)

        # 验证输入
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF文件未找到: {pdf_path}")

        if not pdf_path.suffix.lower() == '.pdf':
            raise ValueError(f"文件必须是PDF格式: {pdf_path}")

        temp_dir = None
        try:
            # 创建临时目录
            temp_dir = tempfile.mkdtemp(prefix='pdf_images_')
            logger.info(f"创建临时目录: {temp_dir}")

            # 将PDF转换为图片
            images = convert_from_path(
                pdf_path,
                dpi=self.dpi,
                output_folder=temp_dir,
                fmt=self.output_format.lower()
            )

            # 获取生成的图片文件列表
            image_files = []
            for i, image in enumerate(images):
                image_filename = f"page_{i+1:03d}.{self.output_format.lower()}"
                image_path = Path(temp_dir) / image_filename
                image.save(image_path, self.output_format)
                image_files.append(str(image_path))
                logger.info(f"保存图片: {image_path}")

            logger.info(f"成功转换 {len(images)} 页为图片")
            return image_files

        except Exception as e:
            logger.error(f"转换PDF为图片时出错: {e}")
            raise
        finally:
            # 注意：这里不清理临时目录，因为调用者需要使用图片
            # 调用者应在完成后调用 cleanup_temp_images()
            pass

    def convert_pdf_to_images_permanent(
        self,
        pdf_path: Union[str, Path],
        output_dir: Union[str, Path],
        filename_prefix: Optional[str] = None
    ) -> List[str]:
        """
        将PDF转换为图片并保存到永久目录。

        参数:
            pdf_path (Union[str, Path]): 输入PDF文件的路径
            output_dir (Union[str, Path]): 保存输出图片的目录
            filename_prefix (Optional[str]): 输出文件名的前缀（默认: PDF文件名）

        返回:
            List[str]: 保存的图片文件路径列表

        异常:
            FileNotFoundError: 如果PDF文件不存在
            ValueError: 如果路径无效
            Exception: 如果转换失败
        """
        pdf_path = Path(pdf_path)
        output_dir = Path(output_dir)

        # 验证输入
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF文件未找到: {pdf_path}")

        if not pdf_path.suffix.lower() == '.pdf':
            raise ValueError(f"文件必须是PDF格式: {pdf_path}")

        # 如果输出目录不存在则创建
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"输出目录: {output_dir}")

        # 确定文件名前缀
        if filename_prefix is None:
            filename_prefix = pdf_path.stem

        try:
            # 将PDF转换为图片
            images = convert_from_path(pdf_path, dpi=self.dpi)

            # 保存图片到输出目录
            saved_files = []
            for i, image in enumerate(images):
                image_filename = f"{filename_prefix}_page_{i+1:03d}.{self.output_format.lower()}"
                image_path = output_dir / image_filename
                image.save(image_path, self.output_format)
                saved_files.append(str(image_path))
                logger.info(f"保存图片: {image_path}")

            logger.info(f"成功转换 {len(images)} 页为图片到 {output_dir}")
            return saved_files

        except Exception as e:
            logger.error(f"转换PDF为图片时出错: {e}")
            raise

    @staticmethod
    def cleanup_temp_images(image_paths: List[str]) -> None:
        """
        清理临时图片文件及其目录。

        参数:
            image_paths (List[str]): 临时图片文件路径列表
        """
        if not image_paths:
            return

        try:
            # 获取第一个图片文件的目录
            first_image_path = Path(image_paths[0])
            temp_dir = first_image_path.parent

            # 删除整个临时目录
            if temp_dir.exists() and temp_dir.name.startswith('pdf_images_'):
                shutil.rmtree(temp_dir)
                logger.info(f"清理临时目录: {temp_dir}")
            else:
                logger.warning(f"目录 {temp_dir} 似乎不是临时目录")

        except Exception as e:
            logger.error(f"清理临时文件时出错: {e}")


# 便捷函数，便于导入和使用
def convert_pdf_to_images_temp(pdf_path: Union[str, Path], dpi: int = 300) -> List[str]:
    """
    临时PDF转图片转换的便捷函数。

    参数:
        pdf_path (Union[str, Path]): 输入PDF文件的路径
        dpi (int): 图片转换的DPI（默认: 300）

    返回:
        List[str]: 临时图片文件路径列表
    """
    converter = PDFToImageConverter(dpi=dpi)
    return converter.convert_pdf_to_images_temp(pdf_path)


def convert_pdf_to_images_permanent(
    pdf_path: Union[str, Path],
    output_dir: Union[str, Path],
    dpi: int = 300,
    filename_prefix: Optional[str] = None
) -> List[str]:
    """
    永久PDF转图片转换的便捷函数。

    参数:
        pdf_path (Union[str, Path]): 输入PDF文件的路径
        output_dir (Union[str, Path]): 保存输出图片的目录
        dpi (int): 图片转换的DPI（默认: 300）
        filename_prefix (Optional[str]): 输出文件名的前缀

    返回:
        List[str]: 保存的图片文件路径列表
    """
    converter = PDFToImageConverter(dpi=dpi)
    return converter.convert_pdf_to_images_permanent(pdf_path, output_dir, filename_prefix)


# 使用示例和测试
def main():
    """
    命令行主函数，处理命令行参数并执行PDF转图片转换。
    """
    parser = argparse.ArgumentParser(
        description="PDF转图片转换器 - 支持临时和永久两种模式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        使用示例:
        # 永久模式 - 保存到指定目录
        python pdf_to_image.py --pdf_path input.pdf --output_path output/
        
        # 永久模式 - 指定DPI和文件名前缀
        python pdf_to_image.py --pdf_path input.pdf --output_path output/ --dpi 600 --prefix converted
        
        # 临时模式 - 只转换不保存
        python pdf_to_image.py --pdf_path input.pdf --temp_mode
        
        # 临时模式 - 指定DPI
        python pdf_to_image.py --pdf_path input.pdf --temp_mode --dpi 300
        """
    )

    # 必需参数
    parser.add_argument(
        '--pdf_path',
        type=str,
        required=True,
        help='输入PDF文件的路径'
    )

    # 模式选择
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        '--output_path',
        type=str,
        help='输出图片的目录路径（永久模式）'
    )
    mode_group.add_argument(
        '--temp_mode',
        action='store_true',
        help='使用临时模式（转换后自动清理）'
    )

    # 可选参数
    parser.add_argument(
        '--dpi',
        type=int,
        default=300,
        help='图片转换的DPI（默认: 300）'
    )

    parser.add_argument(
        '--format',
        type=str,
        default='PNG',
        choices=['PNG', 'JPEG', 'JPG'],
        help='输出图片格式（默认: PNG）'
    )

    parser.add_argument(
        '--prefix',
        type=str,
        help='输出文件名前缀（永久模式，默认使用PDF文件名）'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='显示详细日志信息'
    )

    # 解析参数
    args = parser.parse_args()

    # 设置日志级别
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # 验证PDF文件存在
        pdf_path = Path(args.pdf_path)
        if not pdf_path.exists():
            print(f"错误: PDF文件不存在: {pdf_path}")
            sys.exit(1)

        if not pdf_path.suffix.lower() == '.pdf':
            print(f"错误: 文件必须是PDF格式: {pdf_path}")
            sys.exit(1)

        # 创建转换器
        converter = PDFToImageConverter(dpi=args.dpi, output_format=args.format)

        if args.temp_mode:
            # 临时模式
            print(f"开始转换PDF为图片（临时模式）...")
            print(f"PDF文件: {pdf_path}")
            print(f"DPI: {args.dpi}")
            print(f"格式: {args.format}")

            temp_images = converter.convert_pdf_to_images_temp(pdf_path)

            print(f"\n✅ 成功转换 {len(temp_images)} 页为图片")
            print("临时图片文件:")
            for i, img_path in enumerate(temp_images, 1):
                print(f"  第{i}页: {img_path}")

            print(f"\n⚠️  注意: 这些是临时文件，程序结束后会自动清理")

            # 等待用户确认后清理
            try:
                input("\n按回车键清理临时文件...")
            except KeyboardInterrupt:
                print("\n用户取消，正在清理...")

            converter.cleanup_temp_images(temp_images)
            print("✅ 临时文件已清理")

        else:
            # 永久模式
            output_path = Path(args.output_path)

            print(f"开始转换PDF为图片（永久模式）...")
            print(f"PDF文件: {pdf_path}")
            print(f"输出目录: {output_path}")
            print(f"DPI: {args.dpi}")
            print(f"格式: {args.format}")
            if args.prefix:
                print(f"文件名前缀: {args.prefix}")

            saved_images = converter.convert_pdf_to_images_permanent(
                pdf_path=pdf_path,
                output_dir=output_path,
                filename_prefix=args.prefix
            )

            print(f"\n✅ 成功转换 {len(saved_images)} 页为图片")
            print("保存的图片文件:")
            for i, img_path in enumerate(saved_images, 1):
                print(f"  第{i}页: {img_path}")

    except KeyboardInterrupt:
        print("\n\n用户中断操作")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 转换失败: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
