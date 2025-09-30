import base64
import json
import copy
from tqdm import tqdm
import os
import pandas as pd
from typing import List
import time
from PIL import Image
import math
import os

base_python = r"""
import os
import hashlib
import time
from PIL import Image

def crop(image, crop_box, quality=100):
    # 验证输入坐标
    if len(crop_box) != 4 or any(not 0 <= v <= 1 for v in crop_box):
        raise ValueError("crop_box must be 4 values between 0 and 1")

    try:
        width, height = image.size
        
        # 将归一化坐标转换为实际像素坐标
        x1 = int(crop_box[0] * width)
        y1 = int(crop_box[1] * height)
        x2 = int(crop_box[2] * width)
        y2 = int(crop_box[3] * height)
        
        # 确保坐标在图片范围内
        x1 = max(0, min(x1, width-1))
        y1 = max(0, min(y1, height-1))
        x2 = max(0, min(x2, width-1))
        y2 = max(0, min(y2, height-1))
        
        # 验证裁剪区域有效性
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Invalid cropping area for {x1},{y1} -> {x2},{y2}")
        
        # 裁剪图片
        cropped_img = image.crop((x1, y1, x2, y2))
        return cropped_img
            
    except Exception as e:
        raise IOError(f"Image processing failed: {str(e)}")

def zoom(image, zoom_factor, quality=100):
    # 验证缩放因子
    if zoom_factor <= 0:
        raise ValueError("The scaling factor must be greater than 0.")

    try:
        # 计算新尺寸
        width, height = image.size
        new_size = (int(width * zoom_factor), int(height * zoom_factor))
        
        # 使用LANCZOS高质量缩放
        zoomed_img = image.resize(new_size, Image.LANCZOS)
    
        return zoomed_img
        
    except Exception as e:
        raise IOError(f"Image scaling failed: {str(e)}")
        
def rotate(image, rotate_angle, expand=True, quality=100):
    
    try:
        # 旋转图片（expand=True避免裁剪）
        rotated_img = image.rotate(rotate_angle, expand=expand, resample=Image.BICUBIC)
        
        return rotated_img
            
    except Exception as e:
        raise IOError(f"Image rotation failed: {str(e)}")

def observation(state, save_path=None):
    # 创建文件名映射字典（字母 -> 图片索引）
    image_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}

    # 参数验证
    if len(state) != 4:
        raise ValueError("State must contain 4 elements.")
    
    # 检查是否都是有效字母
    state_chars = [element.upper() for element in state]
    if not all(char in image_map for char in state_chars):
        raise ValueError("State contains invalid characters. Only A, B, C, D are allowed.")
    
    # 检查是否包含所有字母
    if sorted(state_chars) != sorted(['A', 'B', 'C', 'D']):
        raise ValueError("State must contain exactly one of each: A, B, C, D")

    # 加载图片（不旋转）
    ordered_images = []
    for char in state_chars:
        img = image_list[image_map[char]]
        ordered_images.append(img)

    # 获取四张图片的尺寸
    w0, h0 = ordered_images[0].size
    w1, h1 = ordered_images[1].size
    w2, h2 = ordered_images[2].size
    w3, h3 = ordered_images[3].size

    # 计算边距（基于左上角图片尺寸的2%）
    margin_w = int(w0 * 0.02)
    margin_h = int(h0 * 0.02)

    # 计算画布尺寸
    first_row_width = w0 + margin_w + w1
    second_row_width = w2 + margin_w + w3
    canvas_width = max(first_row_width, second_row_width)

    first_row_height = max(h0, h1)
    second_row_height = max(h2, h3)
    canvas_height = first_row_height + margin_h + second_row_height

    # 创建白色背景画布
    canvas = Image.new('RGB', (canvas_width, canvas_height), (255, 255, 255))

    # 定义每个图片的粘贴位置
    positions = [
        (0, 0),  # 左上
        (w0 + margin_w, 0),  # 右上
        (0, first_row_height + margin_h),  # 左下
        (w2 + margin_w, first_row_height + margin_h)  # 右下
    ]

    # 拼接图片
    for img, pos in zip(ordered_images, positions):
        canvas.paste(img, pos)

    return canvas
"""

def read_json(file_path):
    """加载 JSON 文件并返回内容"""
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                print("读取 JSON 文件时出错。")
                return []
    else:
        print("文件不存在。")
        return []
    
def read_jsonl(file_name):
    """
    读取 JSONL 文件并返回包含每一行数据的列表。

    :param file_name: JSONL 文件的名称
    :return: 包含每一行 JSON 对象的列表
    """
    data = []
    with open(file_name, 'r', encoding='utf-8') as file:
        for line in file:
            json_data = json.loads(line)
            data.append(json_data)
    return data

def save_jsonl(data, output_file):
    """
    将列表保存为 JSON Lines 格式的文件。

    参数:
    data (list): 要保存的数据列表，其中每个元素都是一个字典。
    output_file (str): 输出文件的路径。
    """
    with open(output_file, 'w', encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"数据已保存到 {output_file}")

def save_json(data, json_path):
    assert json_path.endswith(".json")
    with open(json_path, 'w', encoding='utf-8') as f_a:
        json.dump(data, f_a, indent=4, ensure_ascii=False)

def diff_dict(dict1, dict2):
    return {k: v for k, v in dict1.items() if k not in dict2 or dict2[k] != v}

def filter_dict(my_dict, key_word):
    return {k: v for k, v in my_dict.items() if key_word in k}

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
