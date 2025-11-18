import os
import cv2
import numpy as np
from PIL import Image
from sam_model import SAMModel
import logging
import argparse
import albumentations as A

# 设置日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def auto_annotate_image(image_path, model, transform=None):
    """
    使用微调后的SAM模型对单张图片进行自动标注
    
    Args:
        image_path: 图片文件路径
        model: SAMModel实例
        transform: 数据增强转换
        
    Returns:
        pred_mask: 预测的二值掩码数组
    """
    try:
        # 加载图片
        try:
            image = Image.open(image_path)
            # 转换为RGB格式
            if image.mode == 'L':  # 灰度图像
                image_rgb = image.convert('RGB')
            elif image.mode == 'RGBA':  # RGBA图像
                image_rgb = image.convert('RGB')
            else:  # RGB或其他颜色模式
                image_rgb = image
            image_rgb = np.array(image_rgb)
        except Exception as e:
            logging.error(f"加载图片失败: {image_path}, 错误: {str(e)}")
            return None
        
        # 调整图片大小
        if transform:
            augmented = transform(image=image_rgb)
            resized_image = augmented['image']
        else:
            resized_image = cv2.resize(image_rgb, (1024, 1024))
        
        # 使用模型预测
        model.set_image(resized_image)
        
        # 创建简单的prompt（使用整个图像作为box）
        h, w = resized_image.shape[:2]
        box = np.array([0, 0, w, h])
        
        # 预测掩码
        masks, _, _ = model.predict(boxes=box, multimask_output=False)
        
        if masks is None or len(masks) == 0:
            logging.error(f"模型预测失败: {image_path}")
            return None
        
        # 获取预测的掩码
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
        
        return pred_mask
        
    except Exception as e:
        logging.error(f"自动标注失败: {image_path}, 错误: {e}")
        import traceback
        traceback.print_exc()
        return None

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
    try:
        sam_model = SAMModel(model_type=args.model_type)
        if not sam_model.load_finetuned_model(args.finetuned_model):
            logging.error("加载微调模型失败")
            return
        logging.info(f"微调模型加载成功: {args.finetuned_model}")
    except Exception as e:
        logging.error(f"模型初始化失败: {e}")
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
        return
    
    logging.info(f"找到 {len(image_files)} 张图片需要标注")
    
    # 遍历图片进行标注
    processed_count = 0
    failed_count = 0
    
    for filename in image_files:
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
        pred_mask = auto_annotate_image(image_path, sam_model, transform)
        
        if pred_mask is not None:
            # 保存掩码为PNG文件
            cv2.imwrite(output_path, (pred_mask * 255).astype(np.uint8))
            processed_count += 1
            logging.info(f"图片标注完成并保存: {output_filename}")
        else:
            failed_count += 1
    
    logging.info(f"自动标注完成！成功标注 {processed_count} 张图片，失败 {failed_count} 张图片")
    logging.info(f"标注结果保存至: {args.output_dir}")

if __name__ == "__main__":
    main()