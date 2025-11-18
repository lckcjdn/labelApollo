#!/usr/bin/env python3
"""
SAM模型微调训练模块
支持使用标注数据对SAM模型进行微调，提高在特定数据集上的性能
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import cv2
import os
import json
import logging
from typing import List, Dict, Tuple, Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2

from segment_anything import sam_model_registry
from sam_model import SAMModel


class SAMFinetuningDataset(Dataset):
    """SAM微调数据集类"""
    
    def __init__(self, data_list: List[Dict], transform=None, image_size=1024):
        """
        初始化数据集
        
        Args:
            data_list: 数据列表，每项包含image_path和mask_path
            transform: 数据增强变换
            image_size: 图像尺寸
        """
        self.data_list = data_list
        self.transform = transform
        self.image_size = image_size
        
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        data_item = self.data_list[idx]
        
        # 加载图像
        image_path = data_item['image_path']
        mask_path = data_item['mask_path']
        
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 加载掩码
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # 确保掩码为二值
        mask = (mask > 0).astype(np.uint8)
        
        # 调整图像尺寸
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        else:
            # 简单的resize
            image = cv2.resize(image, (self.image_size, self.image_size))
            mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        
        return {
            'image': torch.from_numpy(image).permute(2, 0, 1).float() / 255.0,
            'mask': torch.from_numpy(mask).long(),
            'original_size': (image.shape[1], image.shape[0])
        }


class SAMFinetuner:
    """SAM模型微调器"""
    
    def __init__(self, model_type="vit_b", device=None):
        """
        初始化微调器
        
        Args:
            model_type: SAM模型类型
            device: 训练设备
        """
        self.model_type = model_type
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 加载预训练SAM模型
        self.model = sam_model_registry[model_type]()
        
        # 加载checkpoint
        model_path = f"models/sam_{model_type}.pth"
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint)
            logging.info(f"加载预训练模型: {model_path}")
        else:
            raise FileNotFoundError(f"预训练模型文件不存在: {model_path}")
        
        self.model.to(self.device)
        
        # 设置训练模式
        self.model.train()
        
        # 定义损失函数和优化器
        self.criterion = nn.BCEWithLogitsLoss()  # 用于二值分割
        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-5, weight_decay=0.01)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)
        
        logging.info(f"SAM微调器初始化完成，设备: {self.device}")
    
    def prepare_training_data(self, annotated_images: List[Dict]) -> List[Dict]:
        """
        准备训练数据
        
        Args:
            annotated_images: 标注图像列表
            
        Returns:
            处理后的数据列表
        """
        training_data = []
        
        for item in annotated_images:
            try:
                # 获取图像和标注路径
                if isinstance(item, str):
                    # 字符串格式，假设是文件名
                    image_filename = item
                    image_path = os.path.join("uploads", image_filename)
                    
                    # 查找对应的标注文件
                    base_name = os.path.splitext(image_filename)[0]
                    annotation_path = os.path.join("annotations", f"{base_name}.png")
                    
                elif isinstance(item, dict):
                    # 字典格式 - 首先检查是否有image_path和mask_path
                    image_path = item.get('image_path')
                    annotation_path = item.get('mask_path')
                    
                    if not image_path or not annotation_path:
                        # 如果没有提供完整路径，尝试使用旧的逻辑
                        image_path = item.get('filepath') or os.path.join("uploads", item.get('filename', ''))
                        annotation_path = item.get('annotation_path')
                        
                        if not annotation_path:
                            # 如果没有提供标注路径，尝试自动查找
                            filename = item.get('filename', '')
                            base_name = os.path.splitext(filename)[0]
                            annotation_path = os.path.join("annotations", f"{base_name}.png")
                
                # 验证文件存在且是有效的文件
                if image_path and annotation_path:
                    if os.path.isfile(image_path) and os.path.isfile(annotation_path):
                        training_data.append({
                            'image_path': image_path,
                            'mask_path': annotation_path
                        })
                    else:
                        logging.warning(f"文件不存在或不是文件: {image_path} 或 {annotation_path}")
                else:
                    logging.warning(f"无效的路径信息: {image_path} 或 {annotation_path}, 数据项: {item}")
                    
            except Exception as e:
                logging.error(f"处理数据项失败: {e}")
                continue
        
        logging.info(f"准备训练数据: {len(training_data)} 个样本")
        return training_data
    
    def create_data_loader(self, data_list: List[Dict], batch_size=1, shuffle=True):
        """
        创建数据加载器
        
        Args:
            data_list: 数据列表
            batch_size: 批次大小
            shuffle: 是否打乱
            
        Returns:
            DataLoader对象
        """
        # 数据增强
        transform = A.Compose([
            A.Resize(1024, 1024),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        
        dataset = SAMFinetuningDataset(data_list, transform=transform)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2)
    
    def train_epoch(self, data_loader: DataLoader, epoch: int) -> float:
        """
        训练一个epoch
        
        Args:
            data_loader: 数据加载器
            epoch: 当前epoch
            
        Returns:
            平均损失
        """
        total_loss = 0.0
        num_batches = len(data_loader)
        
        for batch_idx, batch in enumerate(data_loader):
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            
            # SAM模型需要特殊的输入格式
            # 这里使用简化的训练方式，实际可能需要更复杂的处理
            with torch.amp.autocast('cuda'):
                # 将图像输入SAM模型
                # 注意：SAM的原始训练可能需要prompt，这里简化处理
                batch_size = images.shape[0]
                losses = []
                
                for i in range(batch_size):
                    single_image = images[i:i+1]
                    single_mask = masks[i:i+1]
                    
                    # 创建简单的prompt（使用整个图像作为box）
                    h, w = single_image.shape[2], single_image.shape[3]
                    boxes = torch.tensor([[[0, 0, w-1, h-1]]], dtype=torch.float32).to(self.device)
                    
                    # 获取模型输出
                    try:
                        # 图像编码：生成当前样本的图像特征
                        image_embeddings = self.model.image_encoder(single_image)
                        
                        # 提示编码：将边界框转换为提示嵌入
                        prompt_embeddings, dense_prompt_embeddings = self.model.prompt_encoder(
                            points=None,
                            boxes=boxes,
                            masks=None
                        )
                        
                        # 掩码解码：生成预测掩码
                        pred_masks, _= self.model.mask_decoder(
                            image_embeddings=image_embeddings,
                            image_pe=self.model.prompt_encoder.get_dense_pe(),
                            sparse_prompt_embeddings=prompt_embeddings,
                            dense_prompt_embeddings=dense_prompt_embeddings,
                            multimask_output=False
                        )
                        
                        # 调整形状并计算损失
                        pred_masks = pred_masks.squeeze(1)  # (1, 256, 256)
                        # 将预测掩码上采样到与目标掩码相同的尺寸(1024x1024)
                        pred_masks = torch.nn.functional.interpolate(
                            pred_masks.unsqueeze(1),  # 添加通道维度 (1, 1, 256, 256)
                            size=(1024, 1024),
                            mode='bilinear',
                            align_corners=False
                        ).squeeze(1)  # 移除通道维度 (1, 1024, 1024)
                        loss = self.criterion(pred_masks, single_mask.float())
                        losses.append(loss)
                    except Exception as e:
                        logging.error(f"批次 {batch_idx} 样本 {i} 训练失败: {e}")
                        import traceback
                        traceback.print_exc()
                        continue
                
                if losses:
                    batch_loss = torch.stack(losses).mean()
                else:
                    continue
                
                # 反向传播
                batch_loss.backward()
                self.optimizer.step()
                
                total_loss += batch_loss.item()
            
            # 打印进度
            if batch_idx % 10 == 0:
                logging.info(f"Epoch {epoch}, Batch {batch_idx}/{num_batches}, Loss: {batch_loss.item():.4f}")
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def finetune(self, annotated_images: List[Dict], epochs=10, batch_size=1, save_path=None):
        """
        执行微调训练
        
        Args:
            annotated_images: 标注图像列表
            epochs: 训练轮数
            batch_size: 批次大小
            save_path: 模型保存路径
            
        Returns:
            训练结果
        """
        logging.info(f"开始SAM模型微调，数据量: {len(annotated_images)}, epochs: {epochs}")
        
        # 准备训练数据
        training_data = self.prepare_training_data(annotated_images)
        
        if len(training_data) == 0:
            raise ValueError("没有可用的训练数据")
        
        # 创建数据加载器
        data_loader = self.create_data_loader(training_data, batch_size=batch_size)
        
        # 训练循环
        train_losses = []
        
        for epoch in range(epochs):
            logging.info(f"开始训练 Epoch {epoch+1}/{epochs}")
            
            # 训练一个epoch
            avg_loss = self.train_epoch(data_loader, epoch)
            train_losses.append(avg_loss)
            
            # 更新学习率
            self.scheduler.step()
            
            logging.info(f"Epoch {epoch+1}/{epochs} 完成, 平均损失: {avg_loss:.4f}")
        
        # 保存微调后的模型
        if save_path is None:
            save_path = f"models/sam_{self.model_type}_finetuned.pth"
        
        torch.save(self.model.state_dict(), save_path)
        logging.info(f"微调模型已保存: {save_path}")
        
        return {
            'epochs': epochs,
            'train_losses': train_losses,
            'final_loss': train_losses[-1] if train_losses else 0.0,
            'model_path': save_path,
            'samples_trained': len(training_data)
        }
    
    def evaluate(self, test_data: List[Dict]) -> Dict:
        """
        评估微调后的模型
        
        Args:
            test_data: 测试数据
            
        Returns:
            评估结果
        """
        self.model.eval()
        
        # 创建测试数据加载器
        test_loader = self.create_data_loader(test_data, batch_size=1, shuffle=False)
        
        total_loss = 0.0
        num_samples = 0
        
        with torch.no_grad():
            for batch in test_loader:
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                
                # 计算损失（简化版本）
                try:
                    h, w = images.shape[2], images.shape[3]
                    boxes = torch.tensor([[[0, 0, w-1, h-1]]], dtype=torch.float32).to(self.device)
                    
                    outputs = self.model(images, boxes=boxes)
                    
                    if isinstance(outputs, dict) and 'masks' in outputs:
                        pred_masks = outputs['masks']
                        loss = self.criterion(pred_masks, masks)
                        total_loss += loss.item()
                        num_samples += 1
                        
                except Exception as e:
                    logging.error(f"评估失败: {e}")
                    continue
        
        avg_loss = total_loss / num_samples if num_samples > 0 else float('inf')
        
        return {
            'test_loss': avg_loss,
            'num_samples': num_samples
        }


def create_sam_finetuning_api():
    """创建SAM微调API"""
    
    def finetune_sam_model(annotated_images, epochs=10, batch_size=2, model_type="vit_l"):
        """
        微调SAM模型
        
        Args:
            annotated_images: 标注图像列表
            epochs: 训练轮数
            batch_size: 批次大小
            model_type: 模型类型
            
        Returns:
            训练结果
        """
        try:
            # 创建微调器
            finetuner = SAMFinetuner(model_type=model_type)
            
            # 执行微调
            result = finetuner.finetune(
                annotated_images=annotated_images,
                epochs=epochs,
                batch_size=batch_size
            )
            
            return {
                'success': True,
                'message': 'SAM模型微调完成',
                'result': result
            }
            
        except Exception as e:
            logging.error(f"SAM微调失败: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    return finetune_sam_model


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    # 解析命令行参数
    import argparse
    parser = argparse.ArgumentParser(description="SAM模型微调脚本")
    parser.add_argument('--images_dir', type=str, required=True, help='图像目录路径')
    parser.add_argument('--annotations_dir', type=str, required=True, help='标注目录路径')
    parser.add_argument('--model_output', type=str, required=True, help='微调后模型输出路径')
    parser.add_argument('--model_type', type=str, default='vit_b', help='模型类型: vit_h, vit_l, vit_b')
    parser.add_argument('--device', type=str, default=None, help='运行设备: cuda或cpu')
    parser.add_argument('--epochs', type=int, default=10, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=2, help='批次大小')
    args = parser.parse_args()
    
    # 创建微调器
    finetuner = SAMFinetuner(model_type=args.model_type, device=args.device)
    
    # 检查输入目录是否存在
    if not os.path.exists(args.images_dir):
        print(f"错误: 图像目录 {args.images_dir} 不存在")
        exit(1)
    if not os.path.exists(args.annotations_dir):
        print(f"错误: 标注目录 {args.annotations_dir} 不存在")
        exit(1)
    
    # 准备训练数据
    annotated_images = []
    for filename in os.listdir(args.images_dir):
        if filename.lower().endswith('.jpg'):
            # 使用绝对路径避免相对路径问题
            image_path = os.path.abspath(os.path.join(args.images_dir, filename))
            # 使用splitext处理扩展名，确保能正确处理.JPG等大写扩展名
            base_name = os.path.splitext(filename)[0]
            annotation_filename = f"{base_name}.png"
            annotation_path = os.path.abspath(os.path.join(args.annotations_dir, annotation_filename))
            annotated_images.append({
                'image_path': image_path,
                'mask_path': annotation_path
            })
    
    print(f"找到 {len(annotated_images)} 个训练样本")
    
    # 执行微调
    try:
        result = finetuner.finetune(annotated_images, epochs=args.epochs, batch_size=args.batch_size, save_path=args.model_output)
        print("微调完成:", result)
    except Exception as e:
        print("微调失败:", str(e))