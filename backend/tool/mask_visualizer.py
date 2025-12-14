#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
掩码图像可视化工具

功能：将掩码图像根据标签定义文件可视化显示
默认使用 Apollo 15类标注配置文件

使用方法：
  python mask_visualizer.py --input /path/to/mask.png --config /path/to/config.yaml --output /path/to/output.png
  python mask_visualizer.py --input /path/to/folder --config /path/to/config.yaml --output /path/to/output/folder
"""

import os
import argparse
import yaml
import cv2
import numpy as np
from glob import glob

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='掩码图像可视化工具')
    parser.add_argument('--input', '-i', required=True, help='输入掩码图像文件或文件夹路径')
    parser.add_argument('--config', '-c', 
                        default='d:\\Ar\\labelApollo\\backend\\vestas_config.yaml', 
                        help='标签定义YAML文件路径，默认为vestas_config.yaml')
    parser.add_argument('--output', '-o', help='输出可视化图像路径，默认为输入路径')
    return parser.parse_args()

def load_config(config_path):
    """加载YAML配置文件"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"错误：无法加载配置文件 {config_path}: {e}")
        return None

def create_color_map(config):
    """根据配置创建颜色映射"""
    # 确保label_colors存在于配置中
    if 'label_colors' not in config:
        print("错误：配置文件中缺少 'label_colors' 部分")
        return None
    
    # 创建标签到颜色的映射
    color_map = {}
    for label, color in config['label_colors'].items():
        if label in config.get('labels', []):
            idx = config['labels'].index(label)
            if label == 'background':
                color_map[0] = color  # 背景ID为0
            else:
                color_map[idx + 1] = color  # 其他标签ID从1开始
    
    return color_map

def visualize_mask(mask_path, color_map, output_path=None):
    """可视化单个掩码图像"""
    try:
        # 读取掩码图像
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"错误：无法读取掩码图像 {mask_path}")
            return False
        
        # 创建彩色图像
        h, w = mask.shape
        color_mask = np.zeros((h, w, 3), dtype=np.uint8)
        
        # 为每个类别设置颜色
        for idx, color in color_map.items():
            # OpenCV使用BGR格式
            bgr_color = (color[2], color[1], color[0])
            color_mask[mask == idx] = bgr_color
        
        # 保存或显示图像
        if output_path:
            # 确保输出目录存在
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            cv2.imwrite(output_path, color_mask)
            print(f"已保存可视化图像到 {output_path}")
        else:
            # 显示图像
            cv2.imshow('Visualized Mask', color_mask)
            print("按任意键继续...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return True
    except Exception as e:
        print(f"错误：处理图像 {mask_path} 时出错: {e}")
        return False

def process_folder(input_folder, color_map, output_folder=None):
    """处理文件夹中的所有掩码图像"""
    # 查找所有PNG图像
    mask_files = glob(os.path.join(input_folder, '*.png'))
    
    if not mask_files:
        print(f"警告：在文件夹 {input_folder} 中未找到PNG图像")
        return
    
    # 如果未指定输出文件夹，使用输入文件夹
    if not output_folder:
        output_folder = input_folder
    
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 处理每个掩码图像
    for mask_file in mask_files:
        # 创建输出文件名
        filename = os.path.basename(mask_file)
        output_path = os.path.join(output_folder, f'visualized_{filename}')
        
        visualize_mask(mask_file, color_map, output_path)

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_arguments()
    
    # 加载配置
    config = load_config(args.config)
    if not config:
        return
    
    # 创建颜色映射
    color_map = create_color_map(config)
    if not color_map:
        return
    
    # 检查输入是文件还是文件夹
    if os.path.isfile(args.input):
        # 处理单个文件
        output_path = args.output
        if not output_path:
            # 默认输出路径：在输入文件名前加上 'visualized_'
            dir_name = os.path.dirname(args.input)
            base_name = os.path.basename(args.input)
            output_path = os.path.join(dir_name, f'visualized_{base_name}')
        
        visualize_mask(args.input, color_map, output_path)
    elif os.path.isdir(args.input):
        # 处理文件夹
        process_folder(args.input, color_map, args.output)
    else:
        print(f"错误：输入路径 {args.input} 既不是文件也不是文件夹")

if __name__ == '__main__':
    main()