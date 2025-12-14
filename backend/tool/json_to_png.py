#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JSON到PNG标注格式转换工具

功能：将LabelMe格式的JSON标注文件转换为掩码图像
默认使用 Apollo 15类标注配置文件

使用方法：
  python json_to_png.py --input /path/to/annotation.json --config /path/to/config.yaml --output /path/to/output.png
  python json_to_png.py --input /path/to/folder --config /path/to/config.yaml --output /path/to/output/folder
"""

import os
import argparse
import json
import yaml
import cv2
import numpy as np
from glob import glob

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='JSON到PNG标注格式转换工具')
    parser.add_argument('--input', '-i', required=True, help='输入JSON标注文件或文件夹路径')
    parser.add_argument('--config', '-c', 
                        default='d:\\Ar\\labelApollo\\backend\\apollo_config.yaml', 
                        help='标签定义YAML文件路径，默认为apollo_config.yaml')
    parser.add_argument('--output', '-o', help='输出掩码图像路径，默认为输入路径')
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

def create_label_to_id_map(config):
    """创建标签到ID的映射"""
    if 'labels' not in config:
        print("错误：配置文件中缺少 'labels' 部分")
        return None
    
    # 创建标签到ID的映射（ID从1开始，0为背景）
    label_to_id = {}
    for idx, label in enumerate(config['labels']):
        label_to_id[label] = idx + 1  # +1 因为背景是0
    
    # 处理通配符标签
    wildcard_mappings = {}
    for label in config['labels']:
        if '*' in label:
            # 将通配符标签存储起来，稍后处理
            wildcard_mappings[label] = label_to_id[label]
    
    return label_to_id, wildcard_mappings

def match_wildcard_label(label, wildcard_mappings):
    """匹配通配符标签"""
    for wildcard_pattern, label_id in wildcard_mappings.items():
        # 将通配符模式转换为正则表达式模式
        pattern_parts = wildcard_pattern.split('*')
        match = True
        for part in pattern_parts:
            if part not in label:
                match = False
                break
        if match:
            return label_id
    return None

def json_to_mask(json_path, label_to_id, wildcard_mappings, output_path=None):
    """将单个JSON文件转换为掩码图像"""
    try:
        # 读取JSON文件
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 获取图像尺寸
        height = data.get('imageHeight')
        width = data.get('imageWidth')
        
        if not height or not width:
            print(f"错误：JSON文件 {json_path} 中缺少图像尺寸信息")
            return False
        
        # 创建空白掩码（0为背景）
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # 处理每个形状
        for shape in data.get('shapes', []):
            label = shape.get('label')
            points = shape.get('points')
            
            if not label or not points:
                continue
            
            # 确定标签ID
            if label in label_to_id:
                label_id = label_to_id[label]
            else:
                # 尝试匹配通配符标签
                label_id = match_wildcard_label(label, wildcard_mappings)
                if label_id is None:
                    print(f"警告：标签 '{label}' 未在配置文件中找到")
                    continue
            
            # 将点转换为整数坐标
            polygon = np.array(points, dtype=np.int32)
            
            # 填充多边形区域
            cv2.fillPoly(mask, [polygon], label_id)
        
        # 保存掩码图像
        if output_path:
            # 确保输出目录存在
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            cv2.imwrite(output_path, mask)
            print(f"已保存掩码图像到 {output_path}")
        else:
            # 显示图像
            cv2.imshow('Mask', mask)
            print("按任意键继续...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return True
    except Exception as e:
        print(f"错误：处理JSON文件 {json_path} 时出错: {e}")
        return False

def process_folder(input_folder, label_to_id, wildcard_mappings, output_folder=None):
    """处理文件夹中的所有JSON文件"""
    # 查找所有JSON文件
    json_files = glob(os.path.join(input_folder, '*.json'))
    
    if not json_files:
        print(f"警告：在文件夹 {input_folder} 中未找到JSON文件")
        return
    
    # 如果未指定输出文件夹，使用输入文件夹
    if not output_folder:
        output_folder = input_folder
    
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 处理每个JSON文件
    for json_file in json_files:
        # 创建输出文件名（将.json替换为.png）
        filename = os.path.basename(json_file)
        output_path = os.path.join(output_folder, filename.replace('.json', '.png'))
        
        json_to_mask(json_file, label_to_id, wildcard_mappings, output_path)

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_arguments()
    
    # 加载配置
    config = load_config(args.config)
    if not config:
        return
    
    # 创建标签到ID的映射
    label_to_id, wildcard_mappings = create_label_to_id_map(config)
    if not label_to_id:
        return
    
    # 检查输入是文件还是文件夹
    if os.path.isfile(args.input):
        # 处理单个文件
        output_path = args.output
        if not output_path:
            # 默认输出路径：将.json替换为.png
            output_path = args.input.replace('.json', '.png')
        
        json_to_mask(args.input, label_to_id, wildcard_mappings, output_path)
    elif os.path.isdir(args.input):
        # 处理文件夹
        process_folder(args.input, label_to_id, wildcard_mappings, args.output)
    else:
        print(f"错误：输入路径 {args.input} 既不是文件也不是文件夹")

if __name__ == '__main__':
    main()
