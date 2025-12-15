import cv2
import numpy as np
import json
import yaml
import argparse
import os
import glob

def mask_to_json(mask_path, classes_path, output_path=None):
    """
    将PNG分割掩码转换为标注工具使用的JSON格式
    
    参数：
    mask_path: PNG掩码文件路径
    classes_path: 类别定义config.yaml文件路径
    output_path: 输出JSON文件路径
    """
    # 检查文件名是否以_mask.png结尾
    if not mask_path.endswith('_mask.png'):
        print(f"跳过文件 {mask_path}：文件名不以_mask.png结尾")
        return None
    
    # 读取类别定义
    with open(classes_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 将labels转换为ID映射字典
    classes = {}
    for i, label in enumerate(config['labels']):
        classes[str(i)] = label
    
    # 读取掩码图像
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"无法读取掩码文件: {mask_path}")
    
    # 获取图像尺寸
    height, width = mask.shape
    
    # 初始化多边形列表
    polygons = []
    
    # 遍历每个类别
    for class_id, class_name in classes.items():
        class_id = int(class_id)
        
        # 获取当前类别的掩码区域
        class_mask = np.where(mask == class_id, 255, 0).astype(np.uint8)
        
        # 查找轮廓
        contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # 简化轮廓
            epsilon = 0.001 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # 将轮廓转换为点列表
            points = []
            for point in approx:
                x, y = point[0]
                points.append({
                    "x": float(x),
                    "y": float(y)
                })
            
            # 仅添加包含至少3个点的多边形
            if len(points) >= 3:
                polygons.append({
                    "id": f"polygon_{len(polygons)}_{class_id}",
                    "points": points,
                    "classId": class_id
                })
    
    # 构建LabelMe兼容的JSON结构
    # 转换为shapes格式
    shapes = []
    for polygon in polygons:
        # 跳过background类别
        if classes[str(polygon["classId"])] == "background":
            continue
            
        shapes.append({
            "label": classes[str(polygon["classId"])],  # 使用类别名称而非ID
            "points": [[p["x"], p["y"]] for p in polygon["points"]],  # 转换为[[x1,y1], [x2,y2], ...]格式
            "group_id": None,
            "shape_type": "polygon",
            "flags": {},
            "description": ""
        })
    
    # 处理图像路径，去掉_mask后缀
    image_path = os.path.basename(mask_path)
    base_name = os.path.splitext(image_path)[0]
    if base_name.endswith('_mask'):
        base_name = base_name[:-5]  # 去掉_mask
    # 保留原始文件扩展名
    image_path = f"{base_name}{os.path.splitext(image_path)[1]}"
    
    annotation_data = {
        "version": "5.0.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": image_path,
        "imageData": None,
        "imageHeight": height,
        "imageWidth": width
    }
    
    # 写入输出文件
    if output_path is None:
        # 去掉_mask后缀
        base_name = os.path.splitext(os.path.basename(mask_path))[0]
        if base_name.endswith('_mask'):
            base_name = base_name[:-5]  # 去掉_mask
        output_path = f"{base_name}.json"
    
    with open(output_path, 'w') as f:
        json.dump(annotation_data, f, indent=2)
    
    print(f"转换完成: {output_path}")
    return output_path

if __name__ == "__main__":
    # 设置命令行参数
    parser = argparse.ArgumentParser(description="将PNG分割掩码转换为JSON标注")
    parser.add_argument("mask_path", help="PNG掩码文件路径或目录")
    parser.add_argument("classes_path", help="类别定义JSON文件路径")
    parser.add_argument("-o", "--output", help="输出目录", default=r"D:\Ar\labelApollo\backend\Artest")
    
    # 解析参数
    args = parser.parse_args()
    
    try:
        # 确保输出目录存在
        os.makedirs(args.output, exist_ok=True)
        
        # 检查mask_path是否为目录
        if os.path.isdir(args.mask_path):
            # 处理目录下所有以_mask.png结尾的文件
            mask_files = []
            for file in os.listdir(args.mask_path):
                if file.endswith('_mask.png') and '_colored_mask.png' not in file:
                    mask_files.append(os.path.join(args.mask_path, file))
            print(f"找到 {len(mask_files)} 个符合条件的掩码文件")
            for mask_file in mask_files:
                # 生成输出文件路径
                base_name = os.path.splitext(os.path.basename(mask_file))[0]
                if base_name.endswith('_mask'):
                    base_name = base_name[:-5]  # 去掉_mask
                output_file = os.path.join(args.output, f"{base_name}.json")
                
                try:
                    mask_to_json(mask_file, args.classes_path, output_file)
                except Exception as e:
                    print(f"处理文件 {mask_file} 失败: {e}")
        else:
            # 处理单个文件
            mask_to_json(args.mask_path, args.classes_path, args.output)
    except Exception as e:
        print(f"转换失败: {e}")
        exit(1)