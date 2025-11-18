#!/usr/bin/env python3
"""
SAM模型微调功能测试脚本
测试微调训练、模型加载和预测功能
"""

import requests
import json
import os
import time
import numpy as np
from PIL import Image, ImageDraw
import base64
import io

class SAMFinetuningTester:
    def __init__(self, base_url="http://127.0.0.1:5000"):
        self.base_url = base_url
        
    def create_test_annotation(self, image_path, annotation_path):
        """创建测试标注数据"""
        # 创建一个简单的测试图像
        image = Image.new('RGB', (512, 512), color='white')
        draw = ImageDraw.Draw(image)
        
        # 绘制一些简单的形状
        draw.rectangle([100, 100, 200, 200], fill='red')
        draw.ellipse([300, 300, 400, 400], fill='blue')
        
        # 保存图像
        image.save(image_path)
        
        # 创建对应的标注掩码
        mask = Image.new('L', (512, 512), color=0)
        mask_draw = ImageDraw.Draw(mask)
        mask_draw.rectangle([100, 100, 200, 200], fill=1)
        mask_draw.ellipse([300, 300, 400, 400], fill=1)
        
        mask.save(annotation_path)
        
        print(f"创建测试数据: {image_path}, {annotation_path}")
        
    def test_health_check(self):
        """测试健康检查"""
        print("=== 测试健康检查 ===")
        try:
            response = requests.get(f"{self.base_url}/api/health")
            if response.status_code == 200:
                data = response.json()
                print(f"✅ 健康检查成功: {data}")
                return True
            else:
                print(f"❌ 健康检查失败: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ 健康检查异常: {e}")
            return False
    
    def test_sam_init(self):
        """测试SAM模型初始化"""
        print("\n=== 测试SAM模型初始化 ===")
        try:
            print("正在发送SAM初始化请求...")
            response = requests.post(
                f"{self.base_url}/api/sam_init",
                json={"model_type": "vit_l"},  # 使用vit_l模型，因为已下载
                timeout=60  # 设置60秒超时
            )
            print(f"收到响应，状态码: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"✅ SAM初始化成功: {data}")
                return True
            else:
                print(f"❌ SAM初始化失败: {response.status_code}, {response.text}")
                return False
        except requests.exceptions.Timeout:
            print(f"❌ SAM初始化超时（60秒），可能是模型加载时间过长")
            return False
        except Exception as e:
            print(f"❌ SAM初始化异常: {e}")
            return False
    
    def test_sam_finetuning(self):
        """测试SAM模型微调"""
        print("\n=== 测试SAM模型微调 ===")
        
        # 创建测试数据
        test_image_path = "uploads/test_finetune_image.jpg"
        test_annotation_path = "annotations/test_finetune_image_annotation.png"
        
        # 确保目录存在
        os.makedirs("uploads", exist_ok=True)
        os.makedirs("annotations", exist_ok=True)
        
        # 创建测试标注数据
        self.create_test_annotation(test_image_path, test_annotation_path)
        
        try:
            # 准备微调数据
            annotated_images = [
                {
                    "filepath": test_image_path,
                    "annotation_path": test_annotation_path
                }
            ]
            
            # 发起微调请求
            print("正在发送微调请求...")
            response = requests.post(
                f"{self.base_url}/api/sam_finetune",
                json={
                    "annotated_images": annotated_images,
                    "epochs": 2,  # 少量epoch用于测试
                    "batch_size": 1,
                    "model_type": "vit_l"  # 使用vit_l模型
                },
                timeout=300  # 设置5分钟超时，微调可能需要更长时间
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ SAM微调成功: {data}")
                return data.get('result', {}).get('model_path')
            else:
                print(f"❌ SAM微调失败: {response.status_code}, {response.text}")
                return None
        except Exception as e:
            print(f"❌ SAM微调异常: {e}")
            return None
    
    def test_load_finetuned_model(self, model_path):
        """测试加载微调后的模型"""
        print(f"\n=== 测试加载微调模型: {model_path} ===")
        
        if not model_path or not os.path.exists(model_path):
            print(f"❌ 微调模型文件不存在: {model_path}")
            return False
        
        try:
            print("正在发送加载微调模型请求...")
            response = requests.post(
                f"{self.base_url}/api/sam_load_finetuned",
                json={
                    "model_path": model_path,
                    "model_type": "vit_l"  # 使用vit_l模型
                },
                timeout=60  # 设置60秒超时
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ 微调模型加载成功: {data}")
                return True
            else:
                print(f"❌ 微调模型加载失败: {response.status_code}, {response.text}")
                return False
        except Exception as e:
            print(f"❌ 微调模型加载异常: {e}")
            return False
    
    def test_sam_prediction(self):
        """测试SAM预测功能"""
        print("\n=== 测试SAM预测功能 ===")
        
        # 使用之前创建的测试图像
        test_image_filename = "test_finetune_image.jpg"
        
        try:
            # 测试点预测
            response = requests.post(
                f"{self.base_url}/api/sam_predict_points",
                json={
                    "image_filename": test_image_filename,
                    "points": [[150, 150], [350, 350]],  # 点击绘制的形状中心
                    "labels": [1, 1],
                    "multimask_output": False
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                results = data.get('results', [])
                print(f"✅ SAM点预测成功: 生成了 {len(results)} 个掩码")
                return True
            else:
                print(f"❌ SAM点预测失败: {response.status_code}, {response.text}")
                return False
        except Exception as e:
            print(f"❌ SAM点预测异常: {e}")
            return False
    
    def run_all_tests(self):
        """运行所有测试"""
        print("开始SAM微调功能完整测试...")
        
        results = {}
        
        # 1. 健康检查
        results['health'] = self.test_health_check()
        if not results['health']:
            print("❌ 服务器不健康，停止测试")
            return results
        
        # 2. SAM初始化
        results['init'] = self.test_sam_init()
        if not results['init']:
            print("❌ SAM初始化失败，停止测试")
            return results
        
        # 3. SAM微调
        model_path = self.test_sam_finetuning()
        results['finetuning'] = model_path is not None
        
        if results['finetuning']:
            # 4. 加载微调模型
            results['load_model'] = self.test_load_finetuned_model(model_path)
            
            # 5. 测试预测
            results['prediction'] = self.test_sam_prediction()
        else:
            results['load_model'] = False
            results['prediction'] = False
        
        # 总结
        print("\n" + "="*50)
        print("测试结果总结:")
        print("="*50)
        
        for test_name, result in results.items():
            status = "✅ 通过" if result else "❌ 失败"
            print(f"{test_name:15} : {status}")
        
        passed = sum(results.values())
        total = len(results)
        print(f"\n总体结果: {passed}/{total} 测试通过")
        
        return results

def main():
    """主函数"""
    tester = SAMFinetuningTester()
    results = tester.run_all_tests()
    
    # 如果所有测试通过，说明SAM微调功能正常工作
    if all(results.values()):
        print("\n🎉 所有测试通过！SAM微调功能已成功实现！")
        print("\n📋 功能说明:")
        print("1. ✅ SAM模型初始化")
        print("2. ✅ 使用标注数据进行微调训练")
        print("3. ✅ 加载微调后的模型")
        print("4. ✅ 使用微调模型进行预测")
        print("\n🚀 您现在可以:")
        print("- 使用标注好的数据训练SAM模型")
        print("- 用微调后的模型标注剩余文件")
        print("- 获得更适合您数据集的分割效果")
    else:
        print("\n⚠️  部分测试失败，请检查错误信息并修复")

if __name__ == "__main__":
    main()