import os
import cv2
import numpy as np
from PIL import Image
from sam_model import SAMModel
import logging
import argparse
import albumentations as A
import json

# 设置日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def auto_annotate_image(image_path, model, transform=None, confidence_threshold=0.8):
    """
    使用微调后的SAM模型对单张图片进行自动标注
    
    Args:
        image_path: 图片文件路径
        model: SAMModel实例
        transform: 数据增强转换
        confidence_threshold: 置信度阈值，用于过滤低置信度的标注结果
        
    Returns:
        pred_mask: 预测的二值掩码数组
        image_info: 图片信息（高度、宽度等）
    """
    print(f"[DEBUG] 开始处理图片: {image_path}")
    try:
        # 加载图片
        try:
            print(f"[DEBUG] 加载图片: {image_path}")
            image = Image.open(image_path)
            # 转换为RGB格式
            if image.mode == 'L':  # 灰度图像
                image_rgb = image.convert('RGB')
            elif image.mode == 'RGBA':  # RGBA图像
                image_rgb = image.convert('RGB')
            else:  # RGB或其他颜色模式
                image_rgb = image
            image_rgb = np.array(image_rgb)
            print(f"[DEBUG] 图片加载成功，形状: {image_rgb.shape}")
        except Exception as e:
            print(f"[ERROR] 加载图片失败: {image_path}, 错误: {str(e)}")
            logging.error(f"加载图片失败: {image_path}, 错误: {str(e)}")
            return None
        
        # 调整图片大小
        print(f"[DEBUG] 调整图片大小")
        if transform:
            augmented = transform(image=image_rgb)
            resized_image = augmented['image']
        else:
            resized_image = cv2.resize(image_rgb, (1024, 1024))
        print(f"[DEBUG] 图片调整后形状: {resized_image.shape}")
        
        try:
            # 使用模型预测
            print(f"[DEBUG] 设置模型图像")
            model.set_image(resized_image)
            
            # 创建简单的prompt（使用整个图像作为box）
            h, w = resized_image.shape[:2]
            box = np.array([0, 0, w, h])
            print(f"[DEBUG] 使用整个图像作为box: {box}")
            
            # 预测掩码
            print(f"[DEBUG] 调用模型预测")
            masks, scores, logits = model.predict(boxes=box, multimask_output=False)
            
            # 过滤低置信度的掩码
            print(f"[DEBUG] 预测结果: masks={masks is not None}, scores={scores}")
            if scores is not None and len(scores) > 0 and scores[0] < confidence_threshold:
                print(f"[DEBUG] 跳过低置信度掩码: 得分={scores[0]}, 阈值={confidence_threshold}")
                logging.info(f"跳过低置信度掩码: {image_path}, 得分: {scores[0]}")
                return None
            
            if masks is None or len(masks) == 0:
                print(f"[ERROR] 模型预测失败: 未生成掩码")
                logging.error(f"模型预测失败: {image_path}")
                return None
            
            # 获取预测的掩码
            print(f"[DEBUG] 获取预测的掩码")
            pred_mask = masks[0]
            
            # 确保是numpy数组
            pred_mask = np.asarray(pred_mask)
            
            # 调试：输出掩码的形状和数据类型
            logging.debug(f"pred_mask shape: {pred_mask.shape}, dtype: {pred_mask.dtype}")
            
            # 确保掩码是2D的
            if pred_mask.ndim == 3:
                # 处理(1, H, W)格式
                if pred_mask.shape[0] == 1:
                    pred_mask = pred_mask[0]
                # 处理(H, W, 1)格式
                elif pred_mask.shape[-1] == 1:
                    pred_mask = np.squeeze(pred_mask, axis=-1)
                # 处理多通道格式，取第一个通道
                else:
                    pred_mask = pred_mask[..., 0]
            
            # 确保掩码是连续的
            pred_mask = np.ascontiguousarray(pred_mask)
            
            # 转换为float32
            pred_mask = pred_mask.astype(np.float32)
            
            # 调试：输出调整后的掩码信息
            logging.debug(f"Resizing mask - shape: {pred_mask.shape}, dtype: {pred_mask.dtype}, contiguous: {pred_mask.flags.contiguous}")
            
            # 将掩码调整回原始图像大小
            pred_mask = cv2.resize(pred_mask, (image_rgb.shape[1], image_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
            
            # 返回掩码和图像信息
            return pred_mask, {
                'height': image_rgb.shape[0],
                'width': image_rgb.shape[1],
                'path': image_path
            }
        finally:
            # 重要：清理图像特征缓存，释放内存
            # 无论预测是否成功，都要确保清理资源
            print("[DEBUG] 清理图像特征缓存")
            try:
                # 检查model是否有reset_image方法（我们刚刚添加的）
                if hasattr(model, 'reset_image'):
                    model.reset_image()
                else:
                    # 如果没有reset_image方法，尝试直接调用predictor的reset_image（如果存在）
                    if hasattr(model, 'predictor') and hasattr(model.predictor, 'reset_image'):
                        model.predictor.reset_image()
            except Exception as e:
                print(f"[WARNING] 清理图像特征缓存失败: {e}")
                logging.warning(f"清理图像特征缓存失败: {e}")
        
    except Exception as e:
        logging.error(f"自动标注失败: {image_path}, 错误: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_labelme_json(image_path, masks, image_info):
    """
    创建符合LabelMe格式的JSON标注文件
    
    Args:
        image_path: 图片文件路径
        masks: 预测的掩码数组
        image_info: 图片信息
        
    Returns:
        labelme_json: LabelMe格式的JSON数据
    """
    # 默认类别为'object'
    default_class = 'object'
    
    # 创建shapes数组
    shapes = []
    
    # 查找掩码中的轮廓
    contours, _ = cv2.findContours(
        (masks * 255).astype(np.uint8), 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    # 为每个轮廓创建一个shape
    for contour in contours:
        if len(contour) >= 3:  # 确保轮廓至少有3个点（形成多边形）
            # 转换为LabelMe需要的点格式
            points = contour.reshape(-1, 2).tolist()
            
            shape = {
                "label": default_class,
                "points": points,
                "group_id": None,
                "shape_type": "polygon",
                "flags": {},
                "description": ""
            }
            shapes.append(shape)
    
    # 创建LabelMe JSON
    labelme_json = {
        "version": "5.0.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": os.path.basename(image_path),
        "imageData": None,  # 不包含图像数据以减小文件大小
        "imageHeight": image_info['height'],
        "imageWidth": image_info['width']
    }
    
    return labelme_json

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='使用微调后的SAM模型自动标注图片')
    # 创建互斥组，支持单张图片或图片目录
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--images_dir', type=str, help='图片目录路径')
    group.add_argument('--image_path', type=str, help='单张图片路径')
    parser.add_argument('--output_dir', type=str, required=True, help='标注结果输出目录路径')
    parser.add_argument('--finetuned_model', '--model_path', type=str, required=True, help='微调后的模型路径')
    parser.add_argument('--model_type', type=str, default='vit_b', help='模型类型: vit_h, vit_l, vit_b')
    parser.add_argument('--confidence_threshold', type=float, default=0.8, help='置信度阈值，用于过滤低置信度的标注结果')
    
    args = parser.parse_args()
    
    # 验证输入路径
    if args.images_dir:
        # 如果路径以 'backend/' 开头且当前不在项目根目录，则去掉前缀
        if args.images_dir.startswith('backend/'):
            adjusted_path = args.images_dir[8:]
            if os.path.exists(adjusted_path):
                args.images_dir = adjusted_path
            elif not os.path.exists(args.images_dir):
                logging.error(f"图片目录不存在: {args.images_dir}")
                return
        elif not os.path.exists(args.images_dir):
            logging.error(f"图片目录不存在: {args.images_dir}")
            return
    elif args.image_path:
        # 如果路径以 'backend/' 开头且当前不在项目根目录，则去掉前缀
        if args.image_path.startswith('backend/'):
            adjusted_path = args.image_path[8:]
            if os.path.exists(adjusted_path):
                args.image_path = adjusted_path
            elif not os.path.exists(args.image_path):
                logging.error(f"图片文件不存在: {args.image_path}")
                return
        elif not os.path.exists(args.image_path):
            logging.error(f"图片文件不存在: {args.image_path}")
            return
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 调整模型路径如果以 'backend/' 开头
    if args.finetuned_model.startswith('backend/'):
        adjusted_path = args.finetuned_model[8:]
        if os.path.exists(adjusted_path):
            args.finetuned_model = adjusted_path
        elif not os.path.exists(args.finetuned_model):
            logging.error(f"模型文件不存在: {args.finetuned_model}")
            return
    
    # 初始化SAM模型
    sam_model = None
    try:
        sam_model = SAMModel(model_type=args.model_type)
        if not sam_model.load_finetuned_model(args.finetuned_model):
            logging.error("加载微调模型失败")
            return
        logging.info(f"微调模型加载成功: {args.finetuned_model}")
    except Exception as e:
        logging.error(f"模型初始化失败: {e}")
        # 确保在失败时释放资源
        if sam_model is not None and hasattr(sam_model, 'clear_cache'):
            sam_model.clear_cache()
        return
    
    # 创建数据增强转换
    transform = A.Compose([
        A.Resize(1024, 1024),
    ])
    
    # 获取所有图片文件
    image_files = []
    if args.images_dir:
        for filename in os.listdir(args.images_dir):
            if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
                image_files.append(filename)
    elif args.image_path:
        image_files.append(os.path.basename(args.image_path))
    
    if not image_files:
        if args.images_dir:
            logging.info(f"图片目录中没有找到图片: {args.images_dir}")
        else:
            logging.error(f"无法处理图片: {args.image_path}")
        # 清理资源
        if sam_model is not None and hasattr(sam_model, 'clear_cache'):
            sam_model.clear_cache()
        return
    
    logging.info(f"找到 {len(image_files)} 张图片需要标注")
    
    # 遍历图片进行标注
    processed_count = 0
    failed_count = 0
    
    try:
        for i, filename in enumerate(image_files):
            # 构建图片路径
            image_path = os.path.join(args.images_dir, filename) if args.images_dir else args.image_path
            
            # 检查是否已经有标注文件
            base_name = os.path.splitext(filename)[0]
            output_filename = f"{base_name}.png"
            output_path = os.path.join(args.output_dir, output_filename)
            
            if os.path.exists(output_path):
                logging.info(f"跳过已存在的标注文件: {output_filename}")
                continue
            
            logging.info(f"正在标注图片: {filename}")
            
            # 自动标注
            result = auto_annotate_image(image_path, sam_model, transform, args.confidence_threshold)
            
            if result is not None:
                pred_mask, image_info = result
                
                # 保存掩码为PNG文件（保持原有功能）
                cv2.imwrite(output_path, (pred_mask * 255).astype(np.uint8))
                
                # 创建LabelMe格式的JSON文件并保存到图像目录
                image_dir = os.path.dirname(image_path)
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                json_filename = f"{base_name}.json"
                json_path = os.path.join(image_dir, json_filename)
                
                # 创建LabelMe JSON数据
                labelme_json = create_labelme_json(image_path, pred_mask, image_info)
                
                # 保存JSON文件
                try:
                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(labelme_json, f, indent=2, ensure_ascii=False)
                    logging.info(f"JSON标注文件已保存到: {json_path}")
                except Exception as e:
                    logging.error(f"保存JSON标注文件失败: {json_path}, 错误: {str(e)}")
                
                processed_count += 1
                logging.info(f"图片标注完成并保存: {output_filename}")
            else:
                failed_count += 1
            
            # 每处理10张图像，清理一次缓存，防止内存累积
            if (i + 1) % 10 == 0:
                print(f"[DEBUG] 每10张图像清理一次缓存")
                logging.info(f"处理了 {i+1} 张图像，执行缓存清理")
                # 检查是否有clear_cache方法
                if hasattr(sam_model, 'clear_cache'):
                    sam_model.clear_cache()
                else:
                    # 如果没有clear_cache方法，尝试直接释放显存
                    try:
                        import torch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            logging.info("执行了torch.cuda.empty_cache()")
                    except:
                        pass
    
    finally:
        # 处理完成后释放所有资源
        print("[DEBUG] 释放模型资源")
        logging.info("释放模型资源")
        if sam_model is not None:
            # 检查是否有clear_cache方法
            if hasattr(sam_model, 'clear_cache'):
                sam_model.clear_cache()
            # 显式删除模型引用，帮助垃圾回收
            sam_model = None
        # 执行Python垃圾回收
        try:
            import gc
            gc.collect()
            logging.info("执行了Python垃圾回收")
        except:
            pass
    
    logging.info(f"自动标注完成！成功标注 {processed_count} 张图片，失败 {failed_count} 张图片")
    logging.info(f"标注结果保存至: {args.output_dir}")

if __name__ == "__main__":
    main()