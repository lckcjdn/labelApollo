import torch
import numpy as np
from PIL import Image
import cv2
import os
from segment_anything import sam_model_registry, SamPredictor
import urllib.request
import logging

class SAMModel:
    def __init__(self, model_type="vit_l", device=None, lazy_load=False):
        """
        初始化SAM模型
        
        Args:
            model_type: 模型类型，可选 "vit_h", "vit_l", "vit_b"
            device: 设备，自动检测GPU可用性
            lazy_load: 是否延迟加载模型（不自动下载）
        """
        self.model_type = model_type
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.predictor = None
        self.model_url = self._get_model_url()
        self.model_path = f"models/sam_{model_type}.pth"
        
        # 确保models目录存在
        os.makedirs("models", exist_ok=True)
        
        # 只有当lazy_load为False时才自动加载模型
        if not lazy_load:
            self._load_model()
    
    def _get_model_url(self):
        """获取模型下载URL"""
        urls = {
            "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
            "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth", 
            "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
        }
        return urls.get(self.model_type, urls["vit_h"])
    
    def _download_model(self, max_retries=3):
        """下载SAM模型，带重试机制"""
        retry_count = 0
        while retry_count < max_retries:
            try:
                logging.info(f"正在下载SAM模型 {self.model_type}... (尝试 {retry_count+1}/{max_retries})")
                # 确保models目录存在，使用绝对路径和异常处理
                models_dir = os.path.abspath(os.path.dirname(self.model_path))
                try:
                    os.makedirs(models_dir, exist_ok=True)
                except PermissionError:
                    logging.error(f"无法创建models目录: {models_dir}，权限不足")
                    # 尝试使用当前用户目录作为备选
                    self.model_path = os.path.join(os.path.expanduser("~"), ".labelapollo", f"sam_{self.model_type}.pth")
                    models_dir = os.path.dirname(self.model_path)
                    os.makedirs(models_dir, exist_ok=True)
                    logging.info(f"使用备选路径: {self.model_path}")
                
                # 使用chunked下载，更好地处理大文件
                with urllib.request.urlopen(self.model_url) as response, open(self.model_path, 'wb') as out_file:
                    total_size = int(response.info().get('Content-Length', 0))
                    downloaded = 0
                    chunk_size = 8192
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        out_file.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            if percent % 20 < 1:  # 每20%进度打印一次
                                print(f"[SAM] 模型下载进度: {percent:.1f}%")
                
                logging.info(f"模型下载完成，保存至: {self.model_path}")
                return True
            except Exception as e:
                retry_count += 1
                logging.error(f"模型下载失败 (尝试 {retry_count}/{max_retries}): {e}")
                if retry_count < max_retries:
                    print(f"[SAM] 尝试重新下载...")
                    time.sleep(2)  # 等待2秒后重试
        return False
    
    def _load_model(self, custom_model_path=None, force_download=False):
        """加载SAM模型，优化错误处理"""
        try:
            print(f"[SAM] 开始加载模型，custom_model_path={custom_model_path}")
            
            # 导入必要的依赖
            import torch
            from segment_anything import sam_model_registry, SamPredictor
            
            # 确定模型路径
            if custom_model_path:
                model_path = custom_model_path
                if not os.path.exists(model_path):
                    raise Exception(f"自定义模型文件不存在: {model_path}")
                print(f"[SAM] 使用自定义模型路径: {model_path}")
            else:
                model_path = self.model_path
                print(f"[SAM] 检查模型文件: {model_path}")
                # 如果模型文件不存在且force_download为True，先下载
                if not os.path.exists(model_path):
                    print(f"[SAM] 模型文件不存在")
                    if force_download:
                        print(f"[SAM] 开始下载模型...")
                        if not self._download_model():
                            # 尝试使用本地已有的其他类型模型作为备选
                            alt_model_types = [t for t in ['vit_b', 'vit_l', 'vit_h'] if t != self.model_type]
                            for alt_type in alt_model_types:
                                alt_path = f"models/sam_{alt_type}.pth"
                                if os.path.exists(alt_path):
                                    print(f"[SAM] 尝试使用备选模型类型: {alt_type}")
                                    model_path = alt_path
                                    self.model_type = alt_type
                                    break
                            else:
                                raise Exception("模型下载失败，且没有找到备选模型")
                    else:
                        raise Exception("模型文件不存在，且未启用强制下载")
                else:
                    print(f"[SAM] 模型文件已存在")
            
            # 检查CUDA是否可用，避免在没有CUDA时尝试使用
            if self.device == 'cuda' and not torch.cuda.is_available():
                print(f"[SAM] CUDA不可用，切换到CPU")
                self.device = 'cpu'
            
            print(f"[SAM] 注册模型类型: {self.model_type}")
            try:
                # 注册并加载模型
                self.model = sam_model_registry[self.model_type]()
            except KeyError as e:
                # 如果请求的模型类型不存在，尝试使用vit_b（最小的模型）
                print(f"[SAM] 模型类型 {self.model_type} 不存在，尝试使用 vit_b")
                self.model_type = 'vit_b'
                # 更新模型路径
                self.model_path = f"models/sam_{self.model_type}.pth"
                # 重新检查模型文件
                if not os.path.exists(self.model_path) and not custom_model_path:
                    if not self._download_model():
                        raise Exception("无法加载任何SAM模型类型")
                model_path = self.model_path
                self.model = sam_model_registry[self.model_type]()
            
            print(f"[SAM] 开始加载state_dict...")
            # 使用try/except处理可能的权重加载问题
            try:
                # 使用map_location避免GPU/CPU不匹配问题
                state_dict = torch.load(model_path, map_location=torch.device(self.device))
                self.model.load_state_dict(state_dict)
            except Exception as e:
                print(f"[SAM] 直接加载state_dict失败，尝试使用strict=False: {e}")
                # 尝试宽松模式加载
                state_dict = torch.load(model_path, map_location=torch.device(self.device))
                self.model.load_state_dict(state_dict, strict=False)
            
            print(f"[SAM] 将模型移动到设备: {self.device}")
            self.model.to(self.device)
            
            # 尝试设置为评估模式以优化内存使用
            self.model.eval()
            
            print(f"[SAM] 创建predictor...")
            self.predictor = SamPredictor(self.model)
            
            # 更新模型路径
            if custom_model_path:
                self.model_path = custom_model_path
            
            print(f"[SAM] SAM模型加载成功，使用设备: {self.device}，模型路径: {model_path}")
            return True
            
        except Exception as e:
            print(f"[SAM] SAM模型加载失败: {e}")
            import traceback
            traceback.print_exc()
            # 清空模型状态
            self.model = None
            self.predictor = None
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
            print(f"[DEBUG] 开始SAM模型预测，points={points is not None}, boxes={boxes is not None}")
            masks, scores, logits = self.predictor.predict(
                point_coords=points,
                point_labels=labels,
                box=boxes,
                multimask_output=multimask_output
            )
            print(f"[DEBUG] SAM模型预测完成，生成了 {len(masks)} 个掩码，最高置信度: {np.max(scores) if scores is not None else None}")
            
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
    
    def is_loaded(self):
        """检查模型是否已加载"""
        return self.model is not None and self.predictor is not None
    
    def reset_image(self):
        """
        清理图像特征缓存，释放内存和显存
        这是SAM模型内存优化的关键步骤，会清除predictor中缓存的图像特征
        """
        try:
            if self.predictor is not None:
                self.predictor.reset_image()
                print("[SAM] 图像特征缓存已清理")
            return True
        except Exception as e:
            logging.error(f"清理图像特征缓存失败: {e}")
            return False
    
    def clear_cache(self):
        """
        清理PyTorch缓存，释放显存
        在GPU环境下特别有用
        """
        try:
            # 先清理图像特征缓存
            self.reset_image()
            
            # 如果在GPU上运行，清理CUDA缓存
            if self.device == 'cuda' and torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()  # 清理跨进程的CUDA缓存
                print("[SAM] CUDA缓存已清理")
            elif self.device == 'cpu':
                # CPU环境下尝试回收Python内存
                import gc
                gc.collect()
                print("[SAM] Python垃圾回收已执行")
            return True
        except Exception as e:
            logging.error(f"清理缓存失败: {e}")
            return False
    
    def __del__(self):
        """
        析构函数，在对象被销毁时释放资源
        """
        try:
            # 清理缓存
            self.clear_cache()
            # 清空模型和predictor引用，帮助垃圾回收
            self.model = None
            self.predictor = None
            print("[SAM] 模型资源已释放")
        except:
            # 析构函数中不要抛出异常
            pass
    
    def load_finetuned_model(self, model_path):
        """
        加载微调后的模型
        
        Args:
            model_path: 微调模型路径
        """
        try:
            # 尝试检测微调模型的类型
            print(f"[SAM] 尝试加载微调模型: {model_path}")
            
            # 首先尝试不同的模型类型
            original_model_type = self.model_type
            for model_type in ['vit_h', 'vit_l', 'vit_b']:
                try:
                    print(f"[SAM] 尝试使用模型类型: {model_type}")
                    self.model_type = model_type
                    
                    # 创建新模型
                    self.model = sam_model_registry[model_type]()
                    
                    # 尝试加载权重，使用宽松模式
                    state_dict = torch.load(model_path, map_location=torch.device(self.device))
                    
                    # 过滤不匹配的权重
                    filtered_state_dict = {k: v for k, v in state_dict.items() 
                                         if k in self.model.state_dict() and 
                                         v.shape == self.model.state_dict()[k].shape}
                    
                    # 加载过滤后的权重
                    self.model.load_state_dict(filtered_state_dict, strict=False)
                    self.model.to(self.device)
                    self.model.eval()
                    self.predictor = SamPredictor(self.model)
                    self.model_path = model_path
                    
                    print(f"[SAM] 微调模型加载成功，使用类型: {model_type}")
                    print(f"[SAM] 成功加载 {len(filtered_state_dict)} 个权重，跳过 {len(state_dict) - len(filtered_state_dict)} 个不匹配权重")
                    logging.info(f"微调模型加载成功: {model_path}, 使用类型: {model_type}")
                    return True
                except Exception as e:
                    print(f"[SAM] 使用 {model_type} 加载失败: {e}")
                    continue
            
            # 如果所有类型都失败，恢复原始类型并抛出异常
            self.model_type = original_model_type
            raise Exception(f"无法使用任何模型类型加载微调模型")
        except Exception as e:
            logging.error(f"微调模型加载失败: {e}")
            return False

# 全局SAM模型实例
sam_model = None

def get_sam_model(model_type="vit_l", device=None, lazy_load=False):
    """
    获取SAM模型实例（单例模式）
    
    Args:
        model_type: 模型类型，可选 "vit_h", "vit_l", "vit_b"
        device: 设备，自动检测GPU可用性
        lazy_load: 是否延迟加载模型（不自动下载）
    
    Returns:
        SAMModel instance
    """
    global sam_model
    if sam_model is None:
        sam_model = SAMModel(model_type=model_type, device=device, lazy_load=lazy_load)
    # 如果提供了设备参数但模型还未加载，重新加载到指定设备
    elif device is not None and sam_model.device != device:
        # 先清理旧模型的资源
        if sam_model is not None:
            sam_model.clear_cache()
        sam_model = SAMModel(model_type=model_type, device=device, lazy_load=lazy_load)
    return sam_model

def release_sam_model():
    """
    释放全局SAM模型实例，清理所有相关资源
    在长时间运行的服务中，可用于定期释放内存
    """
    global sam_model
    if sam_model is not None:
        sam_model.clear_cache()
        sam_model = None
        print("[SAM] 全局模型实例已释放")
        return True
    return False

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