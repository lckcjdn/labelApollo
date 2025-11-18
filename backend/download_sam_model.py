#!/usr/bin/env python3
"""
SAM模型下载脚本
确保完整下载SAM模型文件
"""

import os
import urllib.request
import sys
import hashlib
import time

def calculate_file_hash(filepath):
    """计算文件的SHA256哈希值"""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()

def download_with_progress(url, filepath, expected_size=None):
    """带进度条的文件下载"""
    def progress_hook(block_num, block_size, total_size):
        if total_size > 0:
            percent = min(100, (block_num * block_size * 100) // total_size)
            sys.stdout.write(f"\r下载进度: {percent}% ({block_num * block_size}/{total_size} bytes)")
            sys.stdout.flush()
        else:
            sys.stdout.write(f"\r已下载: {block_num * block_size} bytes")
            sys.stdout.flush()
    
    try:
        print(f"开始下载: {url}")
        print(f"保存到: {filepath}")
        
        # 使用urllib.request下载，支持进度回调
        urllib.request.urlretrieve(url, filepath, progress_hook)
        print("\n下载完成！")
        
        # 检查文件大小
        actual_size = os.path.getsize(filepath)
        print(f"文件大小: {actual_size / (1024*1024):.1f} MB")
        
        if expected_size and abs(actual_size - expected_size) > 1024:  # 允许1KB误差
            print(f"警告: 文件大小不符合预期 (期望: {expected_size} bytes, 实际: {actual_size} bytes)")
            return False
        
        return True
        
    except Exception as e:
        print(f"\n下载失败: {e}")
        if os.path.exists(filepath):
            os.remove(filepath)
        return False

def main():
    """主函数"""
    print("SAM模型下载工具")
    print("=" * 50)
    
    # 模型配置
    models = {
        "vit_h": {
            "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
            "filename": "sam_vit_h.pth",
            "expected_size": 2564550864,  # 约2.4GB
            "description": "SAM ViT-H (最大最准确的模型)"
        },
        "vit_l": {
            "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
            "filename": "sam_vit_l.pth",
            "expected_size": 1249524607,  # 约1.2GB (实际大小)
            "description": "SAM ViT-L (中等大小模型)"
        },
        "vit_b": {
            "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
            "filename": "sam_vit_b.pth",
            "expected_size": 375042383,  # 约375MB
            "description": "SAM ViT-B (最小最快的模型)"
        }
    }
    
    # 确保models目录存在
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    
    print("可用的模型:")
    for i, (model_type, config) in enumerate(models.items(), 1):
        print(f"{i}. {model_type}: {config['description']} ({config['expected_size'] / (1024*1024):.0f} MB)")
    
    # 默认下载vit_h模型
    choice = input("\n选择要下载的模型 (1-3, 默认1): ").strip()
    if not choice:
        choice = "1"
    
    model_keys = list(models.keys())
    try:
        model_index = int(choice) - 1
        if 0 <= model_index < len(model_keys):
            selected_model = model_keys[model_index]
        else:
            print("无效选择，使用默认模型 vit_h")
            selected_model = "vit_h"
    except ValueError:
        print("无效选择，使用默认模型 vit_h")
        selected_model = "vit_h"
    
    model_config = models[selected_model]
    model_path = os.path.join(models_dir, model_config["filename"])
    
    print(f"\n选择的模型: {selected_model}")
    print(f"描述: {model_config['description']}")
    
    # 检查文件是否已存在
    if os.path.exists(model_path):
        print(f"\n模型文件已存在: {model_path}")
        existing_size = os.path.getsize(model_path)
        expected_size = model_config["expected_size"]
        
        print(f"现有文件大小: {existing_size / (1024*1024):.1f} MB")
        print(f"期望文件大小: {expected_size / (1024*1024):.1f} MB")
        
        if existing_size == expected_size:
            print("文件大小匹配，跳过下载")
            
            # 验证文件完整性
            print("验证文件完整性...")
            try:
                import torch
                checkpoint = torch.load(model_path, map_location='cpu')
                print("✓ 模型文件完整性验证成功")
                return True
            except Exception as e:
                print(f"✗ 模型文件验证失败: {e}")
                print("需要重新下载")
        else:
            print("文件大小不匹配，需要重新下载")
            os.remove(model_path)
    
    # 开始下载
    print(f"\n开始下载 {selected_model} 模型...")
    print("这可能需要几分钟时间，请耐心等待...")
    
    start_time = time.time()
    success = download_with_progress(
        model_config["url"], 
        model_path, 
        model_config["expected_size"]
    )
    
    if not success:
        print("下载失败！")
        return False
    
    download_time = time.time() - start_time
    print(f"下载耗时: {download_time:.1f} 秒")
    
    # 验证下载的文件
    print("\n验证下载的文件...")
    try:
        import torch
        checkpoint = torch.load(model_path, map_location='cpu')
        print("✓ 模型文件完整性验证成功")
        print(f"模型包含的键: {list(checkpoint.keys())}")
        
        # 检查模型结构
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            print(f"模型参数数量: {len(state_dict)}")
        
        print(f"\n🎉 {selected_model} 模型下载并验证成功！")
        print(f"文件路径: {model_path}")
        
        return True
        
    except Exception as e:
        print(f"✗ 模型文件验证失败: {e}")
        print("下载的文件可能损坏，请重新下载")
        if os.path.exists(model_path):
            os.remove(model_path)
        return False

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\n现在可以运行测试脚本验证SAM模型功能:")
            print("python test_sam_simple.py")
        else:
            print("\n下载失败，请检查网络连接或稍后重试")
    except KeyboardInterrupt:
        print("\n\n下载被用户中断")
    except Exception as e:
        print(f"\n发生意外错误: {e}")