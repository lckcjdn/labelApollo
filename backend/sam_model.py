import torch
import numpy as np
from PIL import Image
import cv2
import os
from segment_anything import sam_model_registry, SamPredictor
import urllib.request
import logging

class SAMModel:
    def __init__(self, model_type="vit_l", device=None):
        """
        初始化SAM模型
        
        Args:
            model_type: 模型类型，可选 "vit_h", "vit_l", "vit_b"
            device: 设备，自动检测GPU可用性
        """
        self.model_type = model_type
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.predictor = None
        self.model_url = self._get_model_url()
        self.model_path = f"models/sam_{model_type}.pth"
        
        # 确保models目录存在
        os.makedirs("models", exist_ok=True)
        
        self._load_model()
    
    def _get_model_url(self):
        """获取模型下载URL"""
        urls = {
            "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
            "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth", 
            "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
        }
        return urls.get(self.model_type, urls["vit_h"])
    
    def _download_model(self):
        """下载SAM模型"""
        try:
            logging.info(f"正在下载SAM模型 {self.model_type}...")
            urllib.request.urlretrieve(self.model_url, self.model_path)
            logging.info(f"模型下载完成，保存至: {self.model_path}")
            return True
        except Exception as e:
            logging.error(f"模型下载失败: {e}")
            return False
    
    def _load_model(self, custom_model_path=None):
        """加载SAM模型"""
        try:
            print(f"[SAM] 开始加载模型，custom_model_path={custom_model_path}")
            
            # 确定模型路径
            if custom_model_path:
                model_path = custom_model_path
                if not os.path.exists(model_path):
                    raise Exception(f"自定义模型文件不存在: {model_path}")
            else:
                model_path = self.model_path
                print(f"[SAM] 检查模型文件: {model_path}")
                # 如果模型文件不存在，先下载
                if not os.path.exists(model_path):
                    print(f"[SAM] 模型文件不存在，开始下载...")
                    if not self._download_model():
                        raise Exception("模型下载失败")
                else:
                    print(f"[SAM] 模型文件已存在")
            
            print(f"[SAM] 注册模型类型: {self.model_type}")
            # 注册并加载模型
            self.model = sam_model_registry[self.model_type]()
            print(f"[SAM] 开始加载state_dict...")
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"[SAM] 将模型移动到设备: {self.device}")
            self.model.to(self.device)
            print(f"[SAM] 创建predictor...")
            self.predictor = SamPredictor(self.model)
            
            # 更新模型路径
            if custom_model_path:
                self.model_path = custom_model_path
            
            print(f"[SAM] SAM模型加载成功，使用设备: {self.device}，模型路径: {model_path}")
            
        except Exception as e:
            print(f"[SAM] SAM模型加载失败: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def set_image(self, image):
        """
        设置要进行分割的图像
        
        Args:
            image: PIL Image对象或numpy数组
        """
        if isinstance(image, Image.Image):
            image_np = np.array(image)
        else:
            image_np = image
        
        # 转换为RGB格式
        if len(image_np.shape) == 2:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
        elif image_np.shape[2] == 4:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
        
        self.predictor.set_image(image_np)
    
    def predict(self, points=None, boxes=None, labels=None, multimask_output=False):
        """
        进行分割预测
        
        Args:
            points: 点坐标列表，格式为 [[x1, y1], [x2, y2], ...]
            boxes: 边界框列表，格式为 [[x1, y1, x2, y2], ...]
            labels: 点标签列表，1表示前景点，0表示背景点
            multimask_output: 是否输出多个掩码
        
        Returns:
            masks: 分割掩码
            scores: 置信度分数
            logits: 原始输出
        """
        try:
            # 转换为numpy数组
            if points is not None:
                points = np.array(points)
            if boxes is not None:
                boxes = np.array(boxes)
            if labels is not None:
                labels = np.array(labels)
            
            # 执行预测
            masks, scores, logits = self.predictor.predict(
                point_coords=points,
                point_labels=labels,
                box=boxes,
                multimask_output=multimask_output
            )
            
            return masks, scores, logits
            
        except Exception as e:
            logging.error(f"分割预测失败: {e}")
            return None, None, None
    
    def predict_with_points(self, points, labels=None, multimask_output=False):
        """
        使用点进行分割预测
        
        Args:
            points: 点坐标列表
            labels: 点标签列表，默认全部为前景点(1)
            multimask_output: 是否输出多个掩码
        
        Returns:
            masks: 分割掩码
            scores: 置信度分数
        """
        if labels is None:
            labels = [1] * len(points)
        
        masks, scores, _ = self.predict(
            points=points,
            labels=labels,
            multimask_output=multimask_output
        )
        
        return masks, scores
    
    def predict_with_box(self, box, multimask_output=False):
        """
        使用边界框进行分割预测
        
        Args:
            box: 边界框坐标 [x1, y1, x2, y2]
            multimask_output: 是否输出多个掩码
        
        Returns:
            masks: 分割掩码
            scores: 置信度分数
        """
        masks, scores, _ = self.predict(
            boxes=[box],
            multimask_output=multimask_output
        )
        
        return masks, scores
    
    def get_device(self):
        """获取当前使用的设备"""
        return self.device
    
    def get_model_info(self):
        """获取模型信息"""
        return {
            "model_type": self.model_type,
            "device": self.device,
            "model_path": self.model_path,
            "model_loaded": self.model is not None
        }
    
    def load_finetuned_model(self, model_path):
        """
        加载微调后的模型
        
        Args:
            model_path: 微调模型路径
        """
        try:
            self._load_model(custom_model_path=model_path)
            logging.info(f"微调模型加载成功: {model_path}")
            return True
        except Exception as e:
            logging.error(f"微调模型加载失败: {e}")
            return False

# 全局SAM模型实例
sam_model = None

def get_sam_model(model_type="vit_l"):
    """获取SAM模型实例（单例模式）"""
    global sam_model
    if sam_model is None:
        sam_model = SAMModel(model_type=model_type)
    return sam_model

def mask_to_polygon(mask):
    """
    将掩码转换为多边形坐标
    
    Args:
        mask: 二值掩码数组
    
    Returns:
        多边形坐标列表
    """
    try:
        # 使用OpenCV查找轮廓
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        polygons = []
        for contour in contours:
            # 简化轮廓
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # 转换为坐标列表
            if len(approx) >= 3:  # 至少需要3个点
                polygon = approx.flatten().tolist()
                # 转换为 [x1, y1, x2, y2, ...] 格式
                coords = []
                for i in range(0, len(polygon), 2):
                    coords.extend([polygon[i], polygon[i+1]])
                polygons.append(coords)
        
        return polygons
        
    except Exception as e:
        logging.error(f"掩码转多边形失败: {e}")
        return []