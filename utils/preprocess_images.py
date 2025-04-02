# https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Kaggles/DiabeticRetinopathy/preprocess_images.py
"""
尝试移除图像周围不必要的黑色边框，并"裁剪"图像使其占据整个图像区域
"""
import os
import PIL
import cv2
import glob
import warnings
import numpy as np
from tqdm import tqdm
from PIL import Image
from multiprocessing import Pool


def trim(im: Image.Image) -> Image.Image:
    """
    Trim the image to remove black borders.
    Args:
        im: PIL图像对象
    Returns:
        trimmed_image: 裁剪后的PIL图像对象
    """
    percentage = 0.02  # 可调整的参数，用于确定裁剪边界
    img_original = np.array(im)
    img = img_original.copy()

    # Convert to grayscale
    if len(img.shape) == 2:  # 已经是灰度图
        img_gray = img
    elif img.shape[2] == 1:  # 单通道图像
        img_gray = img[:, :, 0]
    else:  # 彩色图像
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    nonzero_pixels = img_gray[img_gray != 0]
    if len(nonzero_pixels) == 0:  # 纯黑图像
        return im

    threshold = 0.1 * np.mean(nonzero_pixels)
    img_binary = img_gray > threshold

    row_sums = np.sum(img_binary, axis=1)
    col_sums = np.sum(img_binary, axis=0)

    rows = np.where(row_sums > img.shape[0] * percentage)[0]
    cols = np.where(col_sums > img.shape[1] * percentage)[0]

    # 计算裁剪边界
    min_row, min_col = np.min(rows), np.min(cols)
    max_row, max_col = np.max(rows), np.max(cols)
    if max_row - min_row < 10 or max_col - min_col < 10:
        return im

    # 裁剪图像
    img_cropped = img_original[min_row:max_row + 1, min_col:max_col + 1]
    return Image.fromarray(img_cropped)


def resize_maintain_aspect(im: Image.Image,
                           target_size: int) -> Image.Image:
    """
    保持纵横比调整图像大小，必要时添加填充。
    Args:
        im: PIL图像对象
        target_size: 目标正方形大小
    Returns:
        resized_image: 调整大小后的PIL图像对象
    """
    width, height = im.size
    ratio = float(target_size) / max(width, height)
    new_width, new_height = int(width * ratio), int(height * ratio)

    # 调整图像大小，注意：ANTIALIAS已被弃用，改用LANCZOS
    im_new = im.resize((new_width, new_height), Image.LANCZOS)

    # 创建新的正方形图像并粘贴调整后的图像
    im_square = Image.new("RGB", (target_size, target_size))
    im_square.paste(im_new, ((target_size - new_width) // 2, (target_size - new_height) // 2))
    return im_square


def save_processed_single_image(args):
    """
    处理单个图像并保存。
    Args:
        args: 包含以下元素的元组:
            - image_name: 图像文件名
            - input_folder_path: 输入图像所在文件夹
            - output_folder_path: 输出图像所在文件夹
            - output_size: 目标大小
    """
    image_name, input_folder_path, output_folder_path, output_size = args
    try:
        input_file_path = os.path.join(input_folder_path, image_name)  # 打开原始图像
        image_new = trim(Image.open(input_file_path))  # 裁剪图像
        image_new = resize_maintain_aspect(image_new, output_size[0])  # 调整图像大小
        output_file_path = os.path.join(output_folder_path, image_name)  # 保存图片
        image_new.save(output_file_path)
    except Exception as e:
        print(f"处理图片 {image_name} 时出错: {str(e)}")


def fast_resize_images(input_folder_path: str,
                       output_folder_path: str,
                       output_size: tuple,
                       file_extensions=None):
    """
    使用多进程处理文件夹中的所有图像。
    Args:
        input_folder_path: 输入图像所在文件夹
        output_folder_path: 输出图像所在文件夹
        output_size: 目标分辨率，例如(150, 150)
        file_extensions: 要处理的文件扩展名，例如['.jpg', '.png']
    """

    # 创建输出文件夹
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    # 默认支持的图像格式
    if file_extensions is None:
        file_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

    # 获取输入文件夹中的所支持的图像文件
    image_files = []
    for ext in file_extensions:
        pattern = os.path.join(input_folder_path, f'*{ext}')
        image_files.extend([os.path.basename(f) for f in glob.glob(pattern)])
        # 支持大写扩展名
        pattern = os.path.join(input_folder_path, f'*{ext.upper()}')
        image_files.extend([os.path.basename(f) for f in glob.glob(pattern)])

    if not image_files:
        print(f"没有找到任何支持的图像文件在 {input_folder_path}")
        return
    print(f"找到 {len(image_files)} 张图片需要处理")

    # 获取CPU核心数量
    num_cores = os.cpu_count()
    print(f"使用 {num_cores} 个CPU核心进行处理")

    # 使用多进程处理图像
    jobs = [(image, input_folder_path, output_folder_path, output_size) for image in image_files]
    with Pool(processes=num_cores) as pool:
        list(tqdm(pool.imap_unordered(save_processed_single_image, jobs), total=len(jobs)))

    print(f"处理完成! 结果保存在 {output_folder_path}")

if __name__ == '__main__':
    # image_original = Image.open(r"../data/kaggle-diabetic-retinopathy-detection/train/10003_left.jpeg")
    # image_original.show()
    # print(image_original.size)

    # image_trimmed = trim(image_original)
    # image_trimmed.show()

    # image_resized_square = resize_maintain_aspect(image_original, target_size=650)
    # image_resized_square.show()

    samples = r'../data/kaggle-diabetic-retinopathy-detection/sample'
    samples_output = os.path.join(samples, 'images_resized_250')
    fast_resize_images(samples, samples_output, output_size=(250, 250))
