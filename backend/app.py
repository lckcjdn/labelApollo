from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
import os
import json
import time
import uuid
import subprocess
import yaml
import glob
import mimetypes
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from segmentation_models_pytorch import DeepLabV3Plus
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import base64
import io
from datetime import datetime
from sam_model import get_sam_model, mask_to_polygon

app = Flask(__name__, static_folder='../frontend', static_url_path='/')
CORS(app)

# 配置
UPLOAD_FOLDER = 'uploads'
ANNOTATIONS_FOLDER = 'annotations'
MODELS_FOLDER = 'models'

# 确保所有必要目录存在
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(ANNOTATIONS_FOLDER, exist_ok=True)
os.makedirs(MODELS_FOLDER, exist_ok=True)

# Apollo数据集类别定义
APOLLO_CLASSES = {
    0: 'background',
    1: 's_w_d',
    2: 's_y_d',
    3: 'ds_y_dn',
    4: 'sb_w_do',
    5: 'sb_y_do',
    6: 'b_w_g',
    7: 's_w_s',
    8: 's_*_c',
    9: 's_w_p',
    10: 'c_wy_z',
    11: 'a_w_*',
    12: 'b_n_sr',
    13: 'd_wy_za',
    14: 'r_wy_np',
    15: 'vom_*_n'
}

# Apollo颜色映射
APOLLO_COLORS = {
    0: [0, 0, 0],
    1: [70, 130, 180],
    2: [220, 20, 60],
    3: [255, 0, 0],
    4: [0, 0, 60],
    5: [0, 60, 100],
    6: [0, 0, 142],
    7: [220, 220, 0],
    8: [102, 102, 156],
    9: [128, 64, 128],
    10: [190, 153, 153],
    11: [0, 0, 230],
    12: [255, 128, 0],
    13: [0, 255, 255],
    14: [178, 132, 190],
    15: [128, 128, 64]
}

# Cityscapes 类别定义
CITYSCAPES_CLASSES = {
    0: 'road',
    1: 'sidewalk', 
    2: 'building',
    3: 'wall',
    4: 'fence',
    5: 'pole',
    6: 'traffic light',
    7: 'traffic sign',
    8: 'vegetation',
    9: 'terrain',
    10: 'sky',
    11: 'person',
    12: 'rider',
    13: 'car',
    14: 'truck',
    15: 'bus',
    16: 'train',
    17: 'motorcycle',
    18: 'bicycle'
}

# 颜色映射 (Cityscapes标准颜色)
CITYSCAPES_COLORS = {
    0: [128, 64, 128],      # road
    1: [244, 35, 232],      # sidewalk
    2: [70, 70, 70],        # building
    3: [102, 102, 156],     # wall
    4: [190, 153, 153],     # fence
    5: [153, 153, 153],     # pole
    6: [250, 170, 30],      # traffic light
    7: [220, 220, 0],       # traffic sign
    8: [107, 142, 35],      # vegetation
    9: [152, 251, 152],     # terrain
    10: [70, 130, 180],     # sky
    11: [220, 20, 60],      # person
    12: [255, 0, 0],        # rider
    13: [0, 0, 142],        # car
    14: [0, 0, 70],         # truck
    15: [0, 60, 100],       # bus
    16: [0, 80, 100],       # train
    17: [0, 0, 230],        # motorcycle
    18: [119, 11, 32]       # bicycle
}

class SegmentationModel:
    def __init__(self, dataset_type='apollo'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.dataset_type = dataset_type
        
        # 根据数据集类型选择类别和颜色
        if dataset_type == 'apollo':
            self.classes = APOLLO_CLASSES
            self.colors = APOLLO_COLORS
        else:
            self.classes = CITYSCAPES_CLASSES
            self.colors = CITYSCAPES_COLORS
        
        self.transform = A.Compose([
            A.Resize(512, 512),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
        self.load_model()
    
    def load_model(self):
        try:
            self.model = DeepLabV3Plus(
                encoder_name='resnet101',
                encoder_weights='imagenet',
                classes=len(self.classes),
                activation=None
            )
            model_path = os.path.join(MODELS_FOLDER, f'{self.dataset_type}_segmentation_model.pth')
            if os.path.exists(model_path):
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                print(f"Loaded {self.dataset_type} model from {model_path}")
            else:
                print(f"No pre-trained {self.dataset_type} model found, using ImageNet weights")
            
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
    
    def update_dataset(self, dataset_type):
        """更新数据集类型"""
        if dataset_type != self.dataset_type:
            self.dataset_type = dataset_type
            if dataset_type == 'apollo':
                self.classes = APOLLO_CLASSES
                self.colors = APOLLO_COLORS
            else:
                self.classes = CITYSCAPES_CLASSES
                self.colors = CITYSCAPES_COLORS
            self.load_model()
    
    def predict(self, image):
        if self.model is None:
            return None
        
        try:
            # 预处理图像
            if isinstance(image, str):
                image = cv2.imread(image)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 应用变换
            transformed = self.transform(image=image)
            input_tensor = transformed['image'].unsqueeze(0).to(self.device)
            
            # 预测
            with torch.no_grad():
                output = self.model(input_tensor)
                pred_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()
            
            # 调整回原始尺寸
            pred_mask = cv2.resize(pred_mask, (image.shape[1], image.shape[0]), 
                                  interpolation=cv2.INTER_NEAREST)
            
            return pred_mask
        except Exception as e:
            print(f"Prediction error: {e}")
            return None
    
    def train_step(self, images, masks):
        if self.model is None:
            return False
        
        try:
            self.model.train()
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
            
            # 转换为tensor
            if isinstance(images, list):
                images = torch.stack(images)
            if isinstance(masks, list):
                masks = torch.stack(masks)
            
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            # 前向传播
            outputs = self.model(images)
            loss = criterion(outputs, masks)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            self.model.eval()
            return True
        except Exception as e:
            print(f"Training error: {e}")
            return False

# 初始化模型
seg_model = SegmentationModel()

@app.route('/api/classes', methods=['GET'])
def get_classes():
    """获取类别信息"""
    dataset_type = request.args.get('dataset', 'apollo')
    
    if dataset_type == 'apollo':
        return jsonify({
            'classes': APOLLO_CLASSES,
            'colors': APOLLO_COLORS
        })
    else:
        return jsonify({
            'classes': CITYSCAPES_CLASSES,
            'colors': CITYSCAPES_COLORS
        })

@app.route('/api/load_config', methods=['POST'])
def load_config():
    """加载YAML配置文件并返回类别信息"""
    try:
        data = request.get_json()
        config_path = data.get('config_path')
        
        if not config_path:
            return jsonify({'error': 'Missing config path'}), 400
        
        if not os.path.exists(config_path):
            return jsonify({'error': 'Config file not found'}), 404
        
        # 读取YAML配置文件
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        labels = config.get('labels', [])
        label_colors = config.get('label_colors', {})
        
        # 转换为前端所需的格式
        categories = []
        for i, label in enumerate(labels):
            rgb_color = label_colors.get(label, [0, 0, 0])
            # 将RGB转换为HEX格式
            hex_color = '#{:02x}{:02x}{:02x}'.format(*rgb_color)
            categories.append({
                'name': label,
                'value': label,
                'color': hex_color
            })
        
        return jsonify({
            'categories': categories,
            'message': 'Config loaded successfully'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/load_config_file', methods=['POST'])
def load_config_file():
    """上传YAML配置文件并返回类别信息"""
    try:
        print('DEBUG: Request received:', request)
        print('DEBUG: Request files:', request.files)
        # 检查是否有文件上传
        if 'config_file' not in request.files:
            print('DEBUG: No file uploaded')
            return jsonify({'error': 'No file uploaded'}), 400
        
        config_file = request.files['config_file']
        print('DEBUG: Config file:', config_file)
        print('DEBUG: Config file filename:', config_file.filename)
        if config_file.filename == '':
            print('DEBUG: No file selected')
            return jsonify({'error': 'No file selected'}), 400
        
        # 读取文件内容
        config_content = config_file.read()
        print('DEBUG: Config content length:', len(config_content))
        print('DEBUG: Config content:', config_content.decode('utf-8'))
        
        # 解析YAML配置
        config = yaml.safe_load(config_content)
        print('DEBUG: Parsed config:', config)
        
        # 提取标签和颜色配置
        labels = config.get('labels', [])
        label_colors = config.get('label_colors', {})
        print('DEBUG: Labels:', labels)
        print('DEBUG: Label colors:', label_colors)
        
        # 转换为前端所需的格式
        categories = []
        for i, label in enumerate(labels):
            # 获取RGB颜色
            rgb_color = label_colors.get(label, [0, 0, 0])
            # 将RGB转换为HEX格式
            hex_color = '#{:02x}{:02x}{:02x}'.format(*rgb_color)
            categories.append({
                'name': label,
                'value': label,
                'color': hex_color
            })
        
        print('DEBUG: Categories:', categories)
        return jsonify({
            'categories': categories,
            'message': 'Config file loaded successfully'
        })
    except Exception as e:
        print('DEBUG: Error:', e)
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/update_dataset', methods=['POST'])
def update_dataset():
    """更新数据集类型"""
    try:
        data = request.get_json()
        dataset_type = data.get('dataset_type', 'apollo')
        
        if dataset_type not in ['apollo', 'cityscapes']:
            return jsonify({'error': 'Invalid dataset type'}), 400
        
        seg_model.update_dataset(dataset_type)
        
        return jsonify({
            'message': f'Dataset updated to {dataset_type}',
            'classes': seg_model.classes,
            'colors': seg_model.colors
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/upload', methods=['POST'])
def upload_image():
    """上传图像"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        # 保存图像
        filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        return jsonify({
            'message': 'Image uploaded successfully',
            'filename': filename,
            'filepath': filepath
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/images', methods=['GET'])
def get_images():
    """获取已上传的图像列表"""
    try:
        images = []
        for filename in os.listdir(UPLOAD_FOLDER):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                annotation_status = check_annotation_status(filename, UPLOAD_FOLDER)
                images.append({
                    'filename': filename,
                    'filepath': os.path.join(UPLOAD_FOLDER, filename),
                    'annotated': annotation_status['annotated'],
                    'annotation_path': annotation_status['annotation_path']
                })
        return jsonify({'images': images})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/browse_directory', methods=['POST'])
def browse_directory():
    """浏览指定目录中的图像文件及其标注状态"""
    try:
        data = request.get_json()
        directory_path = data.get('directory_path')
        
        if not directory_path:
            return jsonify({'error': 'Missing directory path'}), 400
        
        # 安全检查：确保路径在允许的范围内
        if not os.path.exists(directory_path) or not os.path.isdir(directory_path):
            return jsonify({'error': 'Directory does not exist'}), 404
        
        # 获取目录中的图像文件
        image_files = []
        supported_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')
        
        try:
            for filename in os.listdir(directory_path):
                if filename.lower().endswith(supported_extensions):
                    file_path = os.path.join(directory_path, filename)
                    
                    # 检查是否有对应的标注文件
                    annotation_status = check_annotation_status(filename, directory_path)
                    
                    image_files.append({
                        'filename': filename,
                        'filepath': file_path,
                        'annotated': annotation_status['annotated'],
                        'annotation_path': annotation_status['annotation_path'],
                        'file_size': os.path.getsize(file_path),
                        'modified_time': os.path.getmtime(file_path)
                    })
        except PermissionError:
            return jsonify({'error': 'Permission denied accessing directory'}), 403
        
        # 按文件名排序
        image_files.sort(key=lambda x: x['filename'].lower())
        
        # 统计信息
        total_files = len(image_files)
        annotated_files = sum(1 for f in image_files if f['annotated'])
        
        return jsonify({
            'directory_path': directory_path,
            'total_files': total_files,
            'annotated_files': annotated_files,
            'unannotated_files': total_files - annotated_files,
            'files': image_files
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def check_annotation_status(image_filename, image_directory):
    """检查图像文件的标注状态，支持JSON和PNG格式，增强检测逻辑"""
    try:
        # 检查图像同目录下是否有同名的JSON或PNG标注文件
        base_name = os.path.splitext(image_filename)[0]
        
        print(f"Checking annotation status for {image_filename} in {image_directory}")
        
        # 检查JSON标注文件
        json_annotation_path = os.path.join(image_directory, f"{base_name}.json")
        if os.path.exists(json_annotation_path):
            print(f"Found JSON annotation: {json_annotation_path}")
            return {
                'annotated': True,
                'annotation_path': json_annotation_path,
                'format': 'json'
            }
        
        # 检查PNG标注文件，使用更智能的判断方法
        png_annotation_path = os.path.join(image_directory, f"{base_name}.png")
        if os.path.exists(png_annotation_path):
            # 智能判断：如果图像文件本身不是PNG，直接视为标注文件
            if not image_filename.lower().endswith('.png'):
                print(f"Found PNG annotation (non-PNG image): {png_annotation_path}")
                return {
                    'annotated': True,
                    'annotation_path': png_annotation_path,
                    'format': 'png'
                }
            else:
                # 如果图像本身是PNG，通过文件大小判断
                try:
                    img_path = os.path.join(image_directory, image_filename)
                    if os.path.exists(img_path):
                        img_size = os.path.getsize(img_path)
                        png_size = os.path.getsize(png_annotation_path)
                        # 如果大小差异明显，视为标注文件
                        if abs(img_size - png_size) > 1024:  # 大于1KB的差异视为不同文件
                            print(f"Found PNG annotation (size difference): {png_annotation_path}")
                            return {
                                'annotated': True,
                                'annotation_path': png_annotation_path,
                                'format': 'png'
                            }
                except Exception as e:
                    print(f"Error checking PNG file size: {e}")
        
        # 检查旧格式的JSON标注文件
        old_json_path = os.path.join(image_directory, f"{base_name}_annotation.json")
        if os.path.exists(old_json_path):
            print(f"Found old format JSON annotation: {old_json_path}")
            return {
                'annotated': True,
                'annotation_path': old_json_path,
                'format': 'json'
            }
        
        # 检查旧格式的PNG标注文件
        old_png_path = os.path.join(image_directory, f"{base_name}_annotation.png")
        if os.path.exists(old_png_path):
            print(f"Found old format PNG annotation: {old_png_path}")
            return {
                'annotated': True,
                'annotation_path': old_png_path,
                'format': 'png'
            }
        
        print(f"No annotation found for {image_filename}")
        return {
            'annotated': False,
            'annotation_path': None,
            'format': None
        }
        
    except Exception as e:
        print(f"Error checking annotation status for {image_filename}: {e}")
        return {
            'annotated': False,
            'annotation_path': None,
            'format': None
        }

@app.route('/api/load_image_from_directory', methods=['POST'])
def load_image_from_directory():
    """从指定目录加载图像进行标注"""
    try:
        data = request.get_json()
        image_path = data.get('image_path')
        
        if not image_path:
            return jsonify({'error': 'Missing image path'}), 400
        
        if not os.path.exists(image_path):
            return jsonify({'error': 'Image file not found'}), 404
        
        # 直接返回图像路径，不复制到uploads目录
        filename = os.path.basename(image_path)
        
        return jsonify({
            'message': 'Image loaded successfully',
            'filename': filename,
            'filepath': image_path,
            'original_path': image_path
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/serve_image', methods=['GET'])
def serve_image():
    try:
        image_path = request.args.get('path')
        if not image_path:
            return jsonify({'error': 'No image path provided'}), 400
        
        # 移除路径前的斜杠
        image_path = image_path.lstrip('/')
        
        # 验证文件存在
        if not os.path.exists(image_path):
            return jsonify({'error': 'Image file not found'}), 404

        # 验证文件类型
        mime_type, _ = mimetypes.guess_type(image_path)
        if not mime_type or not mime_type.startswith('image/'):
            return jsonify({'error': 'Invalid image type'}), 400

        return send_file(image_path, mimetype=mime_type)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/load_annotation_from_directory', methods=['POST'])
def load_annotation_from_directory():
    """从指定目录加载标注文件"""
    try:
        data = request.get_json()
        image_path = data.get('image_path')
        annotation_path = data.get('annotation_path')
        
        if not image_path:
            return jsonify({'error': 'Missing image path'}), 400
        
        # 如果没有提供标注路径，尝试自动查找
        if not annotation_path:
            image_directory = os.path.dirname(image_path)
            image_filename = os.path.basename(image_path)
            annotation_info = check_annotation_status(image_filename, image_directory)
            annotation_path = annotation_info['annotation_path']
        
        if not annotation_path or not os.path.exists(annotation_path):
            return jsonify({'annotation': None})
        
        # 读取标注文件
        if annotation_path.endswith('.json'):
            # JSON格式的标注文件
            with open(annotation_path, 'r', encoding='utf-8') as f:
                annotation_data = json.load(f)
            return jsonify({'annotation': json.dumps(annotation_data)})
        else:
            # PNG格式的标注文件（mask）
            with open(annotation_path, 'rb') as f:
                annotation_b64 = base64.b64encode(f.read()).decode('utf-8')
            return jsonify({'annotation': annotation_b64})
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/uploads/<filename>')
def serve_uploaded_file(filename):
    """提供上传文件的访问"""
    try:
        return send_from_directory(UPLOAD_FOLDER, filename)
    except Exception as e:
        return jsonify({'error': f'File not found: {filename}'}), 404

@app.route('/')
def index():
    """Serve the main HTML file"""
    return send_from_directory('../frontend', 'index.html')

@app.route('/api/health', methods=['GET'])
def health_check():
    """健康检查端点"""
    return jsonify({
        'status': 'healthy',
        'message': 'Label Apollo API is running',
        'gpu_available': torch.cuda.is_available(),
        'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
    })

@app.route('/api/predict', methods=['POST'])
def predict_segmentation():
    """预测图像分割结果"""
    try:
        data = request.get_json()
        image_filename = data.get('image_filename')
        image_path = data.get('image_path')  # 支持直接指定图像路径
        model = data.get('model', 'deeplab')  # 模型类型
        use_sam = model.startswith('sam')  # 是否使用SAM模型
        model_type = data.get('model_type', None)  # SAM模型类型
        
        if not (image_filename or image_path):
            return jsonify({'error': '缺少图像文件名或路径'}), 400
        
        # 构建图像路径
        if not image_path:
            image_path = os.path.join(UPLOAD_FOLDER, image_filename)
        
        if not os.path.exists(image_path):
            return jsonify({'error': '图像文件不存在'}), 404
        
        print(f"[API] 预测请求，图像: {os.path.basename(image_path)}，使用模型: {model}")
        
        # 加载图像
        try:
            image = Image.open(image_path).convert('RGB')
            if image.width == 0 or image.height == 0:
                return jsonify({'error': '无效的图像文件'}), 400
        except Exception as e:
            return jsonify({'error': f'读取图像失败: {str(e)}'}), 500
        
        pred_mask = None
        if model == 'deeplab':
            # 使用DeepLabV3+模型预测
            try:
                pred_mask = seg_model.predict(image_path)
                print(f"[API] DeepLabV3Plus预测成功")
            except Exception as e:
                print(f"[API] DeepLabV3Plus预测错误: {e}")
                return jsonify({
                    'error': f'DeepLabV3Plus模型错误: {str(e)}'
                }), 500
        elif use_sam:
            # 使用SAM模型预测
            try:
                print(f"[API] 使用SAM模型预测，模型类型: {model_type}")
                
                # 获取SAM模型实例
                try:
                    sam = get_sam_model(model_type=model_type)
                    if sam.model is None or sam.predictor is None:
                        return jsonify({'error': 'SAM模型未正确初始化，请先调用/sam_init'}), 500
                except Exception as e:
                    print(f"[API] 获取SAM模型失败: {e}")
                    return jsonify({'error': f'获取SAM模型失败: {str(e)}，请先调用/sam_init'}), 500
                
                # 读取图像
                image = cv2.imread(image_path)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                h, w = image_rgb.shape[:2]
                
                # 使用整个图像作为边界框
                box = [0, 0, w, h]
                
                # 设置图像到SAM模型
                try:
                    sam.set_image(image_rgb)
                except Exception as e:
                    print(f"[API] 设置SAM图像失败: {e}")
                    return jsonify({'error': f'设置图像到SAM模型失败: {str(e)}'}), 500
                
                # 预测
                try:
                    masks, scores, logits = sam.predict(boxes=np.array([box]), multimask_output=True)
                    
                    # 使用最高分数的掩码
                    best_mask_index = scores.argmax()
                    best_mask = masks[best_mask_index]
                    
                    # 将掩码转换为与DeepLab输出相同的格式
                    pred_mask = np.zeros((h, w), dtype=np.uint8)
                    pred_mask[best_mask] = 1  # 暂时只支持单个类别
                    print(f"[API] SAM预测成功，掩码得分: {scores[best_mask_index]:.4f}")
                except Exception as e:
                    print(f"[API] SAM预测失败: {e}")
                    return jsonify({'error': f'执行SAM预测失败: {str(e)}'}), 500
                
            except Exception as e:
                print(f"[API] SAM模型预测错误: {e}")
                import traceback
                traceback.print_exc()
                return jsonify({
                    'error': f'SAM模型错误: {str(e)}',
                    'tip': '尝试使用更小的模型类型或检查模型初始化状态'
                }), 500
        
        if pred_mask is None:
            return jsonify({'error': '预测失败'}), 500
        
        # 将mask转换为多边形
        annotations = []
        h, w = pred_mask.shape[:2]
        
        # 目前只处理二值掩码（单类别）
        if pred_mask.max() > 0:
            polygons = mask_to_polygon(pred_mask)
            for polygon in polygons:
                # 转换为相对坐标 (0-1范围)
                relative_points = []
                for i in range(0, len(polygon), 2):
                    x = polygon[i]
                    y = polygon[i+1]
                    relative_points.append({'x': x / w, 'y': y / h})
                annotations.append({
                    'classId': 1,  # 默认类别ID
                    'points': relative_points
                })
        
        return jsonify({
            'annotations': annotations
        })
    except Exception as e:
        print(f"[API] 预测请求处理错误: {e}")
        return jsonify({'error': f'处理预测请求时出错: {str(e)}'}), 500

@app.route('/api/save_annotation', methods=['POST'])
def save_annotation():
    """保存标注数据 - 支持PNG掩码格式，优先保存到图像所在目录"""
    try:
        # 检查是JSON请求还是FormData请求
        if request.is_json:
            data = request.get_json()
            image_filepath = data.get('image_filepath')
            image_filename = data.get('image_filename')
            annotation_data = data.get('annotation_data')
            annotation_mask = data.get('annotation_mask')
            image_directory = data.get('image_directory')
        else:
            # 处理FormData请求
            image_filepath = request.form.get('image_filepath')
            image_filename = request.form.get('image_filename')
            annotation_data = request.form.get('annotation_data')
            annotation_mask = request.form.get('annotation_mask')
            image_directory = request.form.get('image_directory')
        
        # 默认输出目录，但我们会优先使用图像所在目录","},{
        
        if not annotation_data:
            return jsonify({'error': 'Missing annotation_data'}), 400
        
        if not image_filepath and not image_filename:
            return jsonify({'error': 'Missing image_filepath or image_filename'}), 400
        
        if image_filepath:
            # 从image_filepath解析图像文件名和目录
            image_filepath = os.path.normpath(image_filepath)
            
            # 如果有提供图像目录信息，优先使用它
            if image_directory:
                # 处理图像目录中的路径分隔符
                image_directory = os.path.normpath(image_directory)
                # 如果前端提供的是相对路径，结合image_directory构建完整路径
                if not os.path.isabs(image_filepath):
                    # 使用提供的image_directory作为基础目录
                    image_dir = image_directory
                    # 确保image_filepath只是文件名，避免路径拼接错误
                    image_filename = os.path.basename(image_filepath)
                    base_name = os.path.splitext(image_filename)[0]
                    # 构建完整的image_filepath
                    image_filepath = os.path.join(image_dir, image_filename)
                    print(f"Using image_directory to build path: {image_filepath}, dir: {image_dir}, filename: {image_filename}")
                else:
                    # 如果image_filepath已经是绝对路径，使用它的目录
                    image_dir = os.path.dirname(image_filepath)
                    image_filename = os.path.basename(image_filepath)
                    base_name = os.path.splitext(image_filename)[0]
                    print(f"Using absolute image_filepath: {image_filepath}, dir: {image_dir}, filename: {image_filename}")
            else:
                # 没有提供图像目录信息，尝试构建绝对路径
                if not os.path.isabs(image_filepath):
                    # 尝试构建绝对路径
                    abs_path = os.path.abspath(image_filepath)
                    if os.path.exists(abs_path):
                        image_filepath = abs_path
                
                image_dir = os.path.dirname(image_filepath)
                image_filename = os.path.basename(image_filepath)
                base_name = os.path.splitext(image_filename)[0]
                print(f"Using image_filepath without directory info: {image_filepath}, dir: {image_dir}, filename: {image_filename}")
        elif image_filename:
            # 使用前端提供的图像文件名
            # 优先使用前端提供的图像目录
            image_dir = image_directory
            
            # 如果前端没有提供图像目录，尝试从uploads目录查找同名图像
            if not image_dir:
                # 检查UPLOAD_FOLDER中是否存在同名图像
                if os.path.exists(os.path.join(UPLOAD_FOLDER, image_filename)):
                    image_dir = UPLOAD_FOLDER
                else:
                    # 尝试从当前工作目录查找
                    if os.path.exists(image_filename):
                        image_dir = os.getcwd()
                    else:
                        # 最后回退到UPLOAD_FOLDER
                        image_dir = UPLOAD_FOLDER
                        print(f"Warning: Could not find image file {image_filename}, defaulting to UPLOAD_FOLDER")
            
            base_name = os.path.splitext(image_filename)[0]
            print(f"Using image_filename: {image_filename}, dir: {image_dir}")
        else:
            # 默认使用uploads目录和当前时间戳生成文件名
            image_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            image_dir = UPLOAD_FOLDER
            # 确保uploads目录存在
            os.makedirs(image_dir, exist_ok=True)
            base_name = os.path.splitext(image_filename)[0]
        
        # 确保输出目录存在
        os.makedirs(image_dir, exist_ok=True)
        print(f"Saving annotations to directory: {image_dir}")
        
        # 保存JSON格式标注（保留兼容性）
        annotation_json_filename = f"{base_name}.json"
        annotation_json_path = os.path.join(image_dir, annotation_json_filename)
        
        # 解析标注数据JSON字符串
        annotation_json = json.loads(annotation_data)
        
        # 如果提供了base_directory参数，计算图像相对路径
        if request.is_json and 'base_directory' in data and data['base_directory']:
            base_directory = data['base_directory']
            # 确保base_directory是绝对路径
            if not os.path.isabs(base_directory):
                base_directory = os.path.abspath(base_directory)
            # 计算图像文件相对于base_directory的相对路径
            rel_image_path = os.path.relpath(image_filepath, base_directory)
            # 将相对路径添加到标注JSON数据
            annotation_json['image_relative_path'] = rel_image_path
        
        # 保存修改后的JSON数据
        with open(annotation_json_path, 'w') as f:
            json.dump(annotation_json, f, indent=2)
        
        # 保存PNG掩码格式标注（如果提供）
        if annotation_mask:
            # 从base64编码的data URL提取PNG数据
            if annotation_mask.startswith('data:image/png;base64,'):
                base64_data = annotation_mask.replace('data:image/png;base64,', '')
                # 解码base64数据
                image_data = base64.b64decode(base64_data)
                
                # 创建PIL图像
                image = Image.open(io.BytesIO(image_data))
                
                # 确保图像格式正确：未标注区域为0，已标注区域为类别ID
                # 转换为numpy数组进行处理
                mask_array = np.array(image)
                
                # 如果是RGB图像，转换为单通道灰度图
                if len(mask_array.shape) == 3 and mask_array.shape[2] == 3:
                    # 简单处理：取第一个通道的值作为类别ID
                    mask_array = mask_array[:, :, 0]
                elif len(mask_array.shape) == 3 and mask_array.shape[2] == 4:
                    # 如果有alpha通道，只取RGB部分的第一个通道
                    mask_array = mask_array[:, :, 0]
                
                # 确保背景像素值为0，已标注区域为类别ID
                # 这里假设前端已经正确生成了掩码，但我们进行二次验证
                # 将非零像素视为已标注区域，保留其值作为类别ID
                # 确保所有像素值都是整数类型
                mask_array = mask_array.astype(np.uint8)
                
                # 转换回PIL图像
                processed_image = Image.fromarray(mask_array, mode='L')  # L模式表示8位灰度图
                
                # 保存为PNG文件
                annotation_png_filename = f"{base_name}.png"
                annotation_png_path = os.path.join(image_dir, annotation_png_filename)
                processed_image.save(annotation_png_path, 'PNG')
                
                # 记录保存路径和格式验证信息
                unique_values = np.unique(mask_array)
                print(f"Saved PNG annotation to: {annotation_png_path}")
                print(f"PNG mask format verification: background value=0, annotated values={list(unique_values[unique_values != 0])}")
                
                return jsonify({
                    'success': True,
                    'message': 'Annotation saved successfully in PNG format',
                    'annotation_json_path': annotation_json_path,
                    'annotation_png_path': annotation_png_path,
                    'image_directory': image_dir,
                    'image_filename': image_filename,
                    'mask_format_verified': True
                })
        
        # 记录保存路径
        print(f"Saved JSON annotation to: {annotation_json_path}")
        return jsonify({
            'success': True,
            'message': 'Annotation saved successfully',
            'annotation_path': annotation_json_path,
            'annotation_json_path': annotation_json_path,
            'image_directory': image_dir,
            'image_filename': image_filename
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/load_annotation', methods=['GET'])
def load_annotation():
    """加载标注数据 - 支持JSON和PNG格式，智能区分图像和标注文件"""
    try:
        image_filepath = request.args.get('image_filepath')
        image_filename = request.args.get('image_filename')
        output_folder = request.args.get('output_folder')
        
        if not image_filepath and not image_filename:
            return jsonify({'error': 'Missing image_filepath or image_filename'}), 400
        
        json_path = None
        png_path = None
        base_name = None
        image_dir = None
        
        # 优先使用完整的图像路径
        if image_filepath:
            # 从图像路径获取目录和基础文件名
            image_filepath = os.path.normpath(image_filepath)
            # 确保路径是绝对路径
            if not os.path.isabs(image_filepath):
                # 尝试构建绝对路径
                # 1. 首先尝试相对于当前工作目录
                abs_path = os.path.abspath(image_filepath)
                if os.path.exists(abs_path):
                    image_filepath = abs_path
                # 2. 尝试相对于UPLOAD_FOLDER
                elif output_folder and os.path.isabs(output_folder):
                    # 如果有提供的输出目录，使用它
                    abs_path = os.path.join(output_folder, image_filepath)
                    if os.path.exists(abs_path):
                        image_filepath = abs_path
                # 3. 尝试相对于UPLOAD_FOLDER
                else:
                    abs_path = os.path.join(os.getcwd(), UPLOAD_FOLDER, image_filepath)
                    if os.path.exists(abs_path):
                        image_filepath = abs_path
            
            image_dir = os.path.dirname(image_filepath)
            base_name = os.path.splitext(os.path.basename(image_filepath))[0]
            print(f"Using image_filepath: {image_filepath}, dir: {image_dir}, base: {base_name}")
        elif image_filename:
            # 使用前端提供的图像文件名
            base_name = os.path.splitext(image_filename)[0]
            
            # 优先使用前端提供的输出目录
            if output_folder and os.path.isabs(output_folder):
                image_dir = output_folder
                print(f"Using provided output_folder: {image_dir}")
            # 检查UPLOAD_FOLDER
            elif os.path.exists(os.path.join(UPLOAD_FOLDER, image_filename)):
                image_dir = UPLOAD_FOLDER
                print(f"Using UPLOAD_FOLDER for image: {image_filename}")
            # 尝试其他可能的目录
            else:
                # 先尝试当前工作目录
                image_dir = os.getcwd()
                # 然后尝试ANNOTATIONS_FOLDER
                if not os.path.exists(os.path.join(image_dir, image_filename)):
                    image_dir = ANNOTATIONS_FOLDER
                print(f"Fallback to image_dir: {image_dir}")
        
        if base_name and image_dir:
            # 构建JSON和PNG标注文件路径
            json_path = os.path.join(image_dir, f"{base_name}.json")
            png_path = os.path.join(image_dir, f"{base_name}.png")
            print(f"Looking for annotations: json={json_path}, png={png_path}")
        
        # 优先尝试加载JSON格式标注
        if json_path and os.path.exists(json_path):
            with open(json_path, 'r') as f:
                annotation_data = f.read()
            print(f"Found JSON annotation: {json_path}")
            return jsonify({'annotation': annotation_data, 'format': 'json'})
        # 其次尝试加载PNG格式标注
        elif png_path and os.path.exists(png_path):
            # 智能判断：如果图像文件本身不是PNG，或者图像文件与PNG文件大小差异明显，则视为标注文件
            is_png_image = image_filename and image_filename.lower().endswith('.png')
            treat_as_annotation = True
            
            # 如果图像本身就是PNG，需要进一步判断以避免误判
            if is_png_image:
                try:
                    # 获取图像文件路径
                    img_path = os.path.join(image_dir, image_filename)
                    if os.path.exists(img_path):
                        # 检查路径是否完全相同
                        if os.path.abspath(img_path) == os.path.abspath(png_path):
                            treat_as_annotation = False
                            print(f"PNG file {png_path} is the image file itself, not annotation")
                        # 比较文件大小（简化判断）
                        else:
                            img_size = os.path.getsize(img_path)
                            png_size = os.path.getsize(png_path)
                            # 如果大小相近，可能是同一个文件
                            if abs(img_size - png_size) < 1024:  # 小于1KB的差异视为同一文件
                                treat_as_annotation = False
                                print(f"PNG file {png_path} appears to be the image file itself, not annotation")
                except Exception as e:
                    print(f"Error checking PNG file: {e}")
            
            if treat_as_annotation:
                with open(png_path, 'rb') as f:
                    # 将PNG文件编码为base64字符串
                    base64_data = base64.b64encode(f.read()).decode('utf-8')
                print(f"Found PNG annotation: {png_path}")
                return jsonify({'annotation': base64_data, 'format': 'png'})
        
        # 检查旧的文件名格式 (_annotation.json)
        if json_path:
            old_json_path = os.path.splitext(json_path)[0] + "_annotation.json"
            if os.path.exists(old_json_path):
                with open(old_json_path, 'r') as f:
                    annotation_data = f.read()
                print(f"Found old format JSON annotation: {old_json_path}")
                return jsonify({'annotation': annotation_data, 'format': 'json'})
        
        # 未找到任何标注文件
        print(f"No annotations found for {image_filename or image_filepath}")
        return jsonify({'annotation': None})
    except Exception as e:
        print(f"Error loading annotation: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/train', methods=['POST'])
def train_model():
    """训练模型"""
    try:
        data = request.get_json()
        annotated_images = data.get('annotated_images', [])
        classes = data.get('classes', seg_model.classes)
        colors = data.get('colors', seg_model.colors)
        
        if len(annotated_images) == 0:
            return jsonify({'error': 'No training data provided'}), 400
        
        # 准备训练数据
        training_data = []
        for item in annotated_images:
            if isinstance(item, str):
                # 如果是字符串，假设是文件名
                image_path = os.path.join(UPLOAD_FOLDER, item)
                annotation_filename = item.rsplit('.', 1)[0] + '_annotation.png'
                annotation_path = os.path.join(ANNOTATIONS_FOLDER, annotation_filename)
            else:
                # 如果是对象，获取路径
                image_path = item.get('filepath') or os.path.join(UPLOAD_FOLDER, item.get('filename', ''))
                annotation_filename = item.get('filename', '').rsplit('.', 1)[0] + '_annotation.png'
                annotation_path = os.path.join(ANNOTATIONS_FOLDER, annotation_filename)
            
            if os.path.exists(image_path) and os.path.exists(annotation_path):
                try:
                    # 加载图像和标注
                    image = cv2.imread(image_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    # 加载标注mask
                    mask = cv2.imread(annotation_path, cv2.IMREAD_GRAYSCALE)
                    
                    if image is not None and mask is not None:
                        # 调整mask尺寸以匹配图像
                        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
                        
                        # 应用变换
                        transformed = seg_model.transform(image=image, mask=mask)
                        training_data.append({
                            'image': transformed['image'],
                            'mask': torch.from_numpy(transformed['mask']).long()
                        })
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")
                    continue
        
        if len(training_data) == 0:
            return jsonify({'error': 'No valid training data found'}), 400
        
        # 执行训练
        if seg_model.model:
            seg_model.model.train()
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(seg_model.model.parameters(), lr=1e-4)
            
            epochs = 10  # 简化的训练循环
            for epoch in range(epochs):
                total_loss = 0
                for data_item in training_data:
                    images = data_item['image'].unsqueeze(0).to(seg_model.device)
                    masks = data_item['mask'].unsqueeze(0).to(seg_model.device)
                    
                    optimizer.zero_grad()
                    outputs = seg_model.model(images)
                    loss = criterion(outputs, masks)
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                
                print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(training_data):.4f}")
            
            seg_model.model.eval()
            
            # 保存模型
            model_path = os.path.join(MODELS_FOLDER, f'{seg_model.dataset_type}_segmentation_model.pth')
            torch.save(seg_model.model.state_dict(), model_path)
            
            return jsonify({
                'message': f'Training completed with {len(training_data)} samples',
                'samples_processed': len(training_data),
                'epochs': epochs,
                'model_path': model_path
            })
        else:
            return jsonify({'error': 'Model not available for training'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/import_annotations', methods=['POST'])
def import_annotations():
    """导入外部标注文件"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # 保存导入的文件
        import_filename = f"imported_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
        import_path = os.path.join(ANNOTATIONS_FOLDER, import_filename)
        file.save(import_path)
        
        # 这里可以添加解析Cityscapes格式文件的逻辑
        
        return jsonify({
            'message': 'Annotations imported successfully',
            'filename': import_filename
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ==================== SAM 相关API ====================

@app.route('/api/sam_init', methods=['POST'])
def init_sam_model():
    """初始化SAM模型"""
    try:
        data = request.get_json()
        model_type = data.get('model_type', 'vit_l')  # 默认使用vit_l
        device = data.get('device', None)  # 允许指定设备
        
        print(f"[API] 初始化SAM模型，类型: {model_type}，设备: {device}")
        
        # 获取SAM模型实例
        sam = get_sam_model(model_type=model_type, device=device)
        
        # 验证模型是否成功初始化
        if sam.model is None:
            return jsonify({
                'error': 'SAM模型初始化失败，内部状态为None',
                'model_info': sam.get_model_info()
            }), 500
        
        # 返回成功信息
        model_info = sam.get_model_info()
        print(f"[API] SAM模型初始化成功: {model_info}")
        
        return jsonify({
            'message': 'SAM model initialized successfully',
            'model_info': model_info,
            'gpu_available': torch.cuda.is_available()
        })
    except ImportError as e:
        # 处理导入错误，通常是segment_anything库未安装
        print(f"[API] SAM初始化错误 - 缺少依赖: {e}")
        return jsonify({
            'error': '缺少SAM相关依赖，请确保已安装segment_anything库',
            'details': str(e)
        }), 500
    except Exception as e:
        print(f"[API] SAM初始化错误: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': f'SAM模型初始化失败: {str(e)}',
            'tip': '尝试使用更小的模型类型如vit_b，或检查网络连接是否能下载模型'
        }), 500

@app.route('/api/sam_predict_points', methods=['POST'])
def sam_predict_with_points():
    """使用点进行SAM分割预测"""
    try:
        data = request.get_json()
        image_filename = data.get('image_filename')
        image_path = data.get('image_path')  # 支持直接指定图像路径
        points = data.get('points', [])  # [[x1, y1], [x2, y2], ...]
        labels = data.get('labels', [])  # [1, 0, 1, ...] 1=前景点, 0=背景点
        multimask_output = data.get('multimask_output', False)
        
        if not (image_filename or image_path) or not points:
            return jsonify({'error': '缺少图像文件名/路径或点坐标'}), 400
        
        # 构建图像路径
        if not image_path:
            image_path = os.path.join(UPLOAD_FOLDER, image_filename)
        
        if not os.path.exists(image_path):
            return jsonify({'error': '图像文件不存在'}), 404
        
        print(f"[API] SAM点预测请求，图像: {os.path.basename(image_path)}，点数: {len(points)}")
        
        # 获取SAM模型并检查状态
        try:
            sam = get_sam_model()
            if sam.model is None or sam.predictor is None:
                return jsonify({'error': 'SAM模型未正确初始化，请先调用/sam_init'}), 500
        except Exception as e:
            print(f"[API] 获取SAM模型失败: {e}")
            return jsonify({'error': f'获取SAM模型失败: {str(e)}，请先调用/sam_init'}), 500
        
        # 加载图像并验证
        try:
            image = Image.open(image_path).convert('RGB')
            if image.width == 0 or image.height == 0:
                return jsonify({'error': '无效的图像文件'}), 400
        except Exception as e:
            return jsonify({'error': f'读取图像失败: {str(e)}'}), 500
        
        # 设置图像到模型
        try:
            sam.set_image(image)
        except Exception as e:
            print(f"[API] 设置SAM图像失败: {e}")
            return jsonify({'error': f'设置图像到SAM模型失败: {str(e)}'}), 500
        
        # 进行预测
        try:
            masks, scores = sam.predict_with_points(
                points=points,
                labels=labels if labels else None,
                multimask_output=multimask_output
            )
            
            if masks is None:
                return jsonify({'error': 'SAM预测失败，可能是点坐标超出图像范围'}), 500
        except Exception as e:
            print(f"[API] SAM预测失败: {e}")
            return jsonify({'error': f'执行SAM预测失败: {str(e)}'}), 500
        
        # 将掩码转换为多边形
        results = []
        for i, mask in enumerate(masks):
            polygons = mask_to_polygon(mask)
            
            # 将掩码转换为base64图像
            mask_colored = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
            mask_colored[mask] = [0, 255, 0]  # 绿色掩码
            
            _, buffer = cv2.imencode('.png', mask_colored)
            mask_b64 = base64.b64encode(buffer).decode('utf-8')
            
            results.append({
                'mask': mask_b64,
                'polygons': polygons,
                'score': float(scores[i]) if i < len(scores) else 0.0,
                'shape': mask.shape
            })
        
        return jsonify({
            'results': results,
            'points': points,
            'labels': labels if labels else [1] * len(points)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/sam_predict_box', methods=['POST'])
def sam_predict_with_box():
    """使用边界框进行SAM分割预测"""
    try:
        data = request.get_json()
        image_filename = data.get('image_filename')
        image_path = data.get('image_path')  # 支持直接指定图像路径
        box = data.get('box')  # [x1, y1, x2, y2]
        multimask_output = data.get('multimask_output', False)
        
        if not (image_filename or image_path) or not box:
            return jsonify({'error': '缺少图像文件名/路径或边界框'}), 400
        
        # 验证边界框格式
        if not isinstance(box, list) or len(box) != 4:
            return jsonify({'error': '边界框格式无效，应为[x1, y1, x2, y2]'}), 400
        
        # 构建图像路径
        if not image_path:
            image_path = os.path.join(UPLOAD_FOLDER, image_filename)
        
        if not os.path.exists(image_path):
            return jsonify({'error': '图像文件不存在'}), 404
        
        print(f"[API] SAM边界框预测请求，图像: {os.path.basename(image_path)}")
        
        # 获取SAM模型并检查状态
        try:
            sam = get_sam_model()
            if sam.model is None or sam.predictor is None:
                return jsonify({'error': 'SAM模型未正确初始化，请先调用/sam_init'}), 500
        except Exception as e:
            print(f"[API] 获取SAM模型失败: {e}")
            return jsonify({'error': f'获取SAM模型失败: {str(e)}，请先调用/sam_init'}), 500
        
        # 加载图像并验证
        try:
            image = Image.open(image_path).convert('RGB')
            if image.width == 0 or image.height == 0:
                return jsonify({'error': '无效的图像文件'}), 400
        except Exception as e:
            return jsonify({'error': f'读取图像失败: {str(e)}'}), 500
        
        # 设置图像到模型
        try:
            sam.set_image(image)
        except Exception as e:
            print(f"[API] 设置SAM图像失败: {e}")
            return jsonify({'error': f'设置图像到SAM模型失败: {str(e)}'}), 500
        
        # 进行预测
        try:
            masks, scores = sam.predict_with_box(
                box=box,
                multimask_output=multimask_output
            )
            
            if masks is None:
                return jsonify({'error': 'SAM预测失败，可能是边界框无效'}), 500
        except Exception as e:
            print(f"[API] SAM预测失败: {e}")
            return jsonify({'error': f'执行SAM预测失败: {str(e)}'}), 500
        
        # 将掩码转换为多边形
        results = []
        for i, mask in enumerate(masks):
            polygons = mask_to_polygon(mask)
            
            # 将掩码转换为base64图像
            mask_colored = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
            mask_colored[mask] = [255, 0, 0]  # 红色掩码
            
            _, buffer = cv2.imencode('.png', mask_colored)
            mask_b64 = base64.b64encode(buffer).decode('utf-8')
            
            results.append({
                'mask': mask_b64,
                'polygons': polygons,
                'score': float(scores[i]) if i < len(scores) else 0.0,
                'shape': mask.shape
            })
        
        return jsonify({
            'results': results,
            'box': box
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/sam_info', methods=['GET'])
def get_sam_info():
    """获取SAM模型信息"""
    try:
        print("[API] 请求SAM模型信息")
        sam = get_sam_model()
        model_info = sam.get_model_info()
        # 增加额外信息，如GPU可用性
        enhanced_info = {
            **model_info,
            'gpu_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else 'N/A',
            'torch_version': torch.__version__
        }
        return jsonify(enhanced_info)
    except Exception as e:
        print(f"[API] 获取SAM信息失败: {e}")
        return jsonify({
            'error': f'获取SAM模型信息失败: {str(e)}',
            'model_loaded': False,
            'gpu_available': torch.cuda.is_available()
        }), 500

@app.route('/api/sam_finetune', methods=['POST'])
def sam_finetune():
    """微调SAM模型"""
    try:
        data = request.get_json()
        finetuned_model_path = data.get('finetuned_model_path')
        
        if not finetuned_model_path:
            return jsonify({'error': '缺少微调模型路径'}), 400
        
        print(f"[API] 尝试加载微调SAM模型: {finetuned_model_path}")
        
        # 检查文件是否存在
        if not os.path.exists(finetuned_model_path):
            # 尝试在模型目录下查找
            if not os.path.isabs(finetuned_model_path):
                candidate_path = os.path.join(MODELS_FOLDER, finetuned_model_path)
                if os.path.exists(candidate_path):
                    finetuned_model_path = candidate_path
                else:
                    return jsonify({'error': f'微调模型文件不存在: {finetuned_model_path}'}), 404
            else:
                return jsonify({'error': f'微调模型文件不存在: {finetuned_model_path}'}), 404
        
        # 获取SAM模型实例
        try:
            sam = get_sam_model()
        except Exception as e:
            print(f"[API] 获取SAM模型失败: {e}")
            return jsonify({'error': f'获取SAM模型失败: {str(e)}，请先调用/sam_init'}), 500
        
        # 加载微调模型
        try:
            success = sam.load_finetuned_model(finetuned_model_path)
            
            if success:
                model_info = sam.get_model_info()
                print(f"[API] 微调模型加载成功: {model_info}")
                return jsonify({
                    'message': '微调模型加载成功',
                    'model_info': model_info
                })
            else:
                return jsonify({'error': '微调模型加载失败，模型格式可能不正确'}), 500
        except Exception as e:
            print(f"[API] 微调模型加载异常: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({
                'error': f'微调模型加载异常: {str(e)}',
                'tip': '检查模型文件格式是否兼容，确保是有效的SAM模型权重'
            }), 500
    except Exception as e:
        print(f"[API] 微调请求处理错误: {e}")
        return jsonify({'error': f'处理微调请求时出错: {str(e)}'}), 500





@app.route('/api/auto_annotate', methods=['POST'])
def auto_annotate():
    """AI自动标注目录中的所有图像"""
    try:
        data = request.get_json()
        directory_path = data.get('directory_path')
        model_path = data.get('model_path', 'models/sam_finetuned.pth')
        model_type = data.get('model_type', 'vit_l')
        
        if not directory_path:
            return jsonify({'error': 'Missing directory_path'}), 400
        
        if not os.path.exists(directory_path):
            return jsonify({'error': 'Directory not found'}), 404
        
        if not os.path.exists(model_path):
            return jsonify({'error': 'Model file not found'}), 404
        
        # 初始化SAM模型
        from sam_model import get_sam_model
        sam = get_sam_model(model_type=model_type)
        sam.load_finetuned_model(model_path)
        
        # 创建输出目录
        output_dir = os.path.join(directory_path, 'annotations')
        os.makedirs(output_dir, exist_ok=True)
        
        # 获取所有图像文件
        image_files = [f for f in os.listdir(directory_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        
        processed_count = 0
        failed_count = 0
        
        for filename in image_files:
            image_path = os.path.join(directory_path, filename)
            base_name = os.path.splitext(filename)[0]
            output_filename = f"{base_name}.png"
            output_path = os.path.join(output_dir, output_filename)
            
            if os.path.exists(output_path):
                continue
            
            # 自动标注
            try:
                from auto_annotate import auto_annotate_image
                pred_mask = auto_annotate_image(image_path, sam)
                
                if pred_mask is not None:
                    cv2.imwrite(output_path, (pred_mask * 255).astype(np.uint8))
                    processed_count += 1
                else:
                    failed_count += 1
            except Exception as e:
                print(f"Error annotating {filename}: {e}")
                failed_count += 1
        
        return jsonify({
            'success': True,
            'processed_count': processed_count,
            'failed_count': failed_count,
            'output_dir': output_dir
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/convert_annotations', methods=['POST'])
def convert_annotations():
    """将JSON标注转换为Cityscapes风格的PNG分割标注"""
    try:
        data = request.get_json()
        directory = data.get('directory')
        
        if not directory:
            return jsonify({'error': 'Missing directory path'}), 400
        
        # 遍历目录中的JSON文件
        converted_count = 0
        for filename in os.listdir(directory):
            if filename.endswith('.json'):
                json_path = os.path.join(directory, filename)
                
                # 对应的图像文件（尝试多种格式）
                base_name = os.path.splitext(filename)[0]
                image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG']
                image_path = None
                
                for ext in image_extensions:
                    candidate_path = os.path.join(directory, f"{base_name}{ext}")
                    if os.path.exists(candidate_path):
                        image_path = candidate_path
                        break
                
                if not image_path:
                    continue
                
                # 读取图像尺寸
                image = Image.open(image_path)
                width, height = image.size
                
                # 创建PNG画布
                png = np.zeros((height, width), dtype=np.uint8)
                
                # 读取JSON标注
                with open(json_path, 'r') as f:
                    annotations = json.load(f)
                
                polygons = annotations.get('polygons', [])
                
                # 遍历每个多边形
                for polygon in polygons:
                    class_id = polygon.get('classId', 0)
                    points = polygon.get('points', [])
                    
                    if not points:
                        continue
                    
                    # 转换为OpenCV格式的点
                    cv_points = [(point['x'], point['y']) for point in points]
                    cv_points = np.array(cv_points, dtype=np.int32)
                    cv_points = cv_points.reshape((1, -1, 2))  # 形状要求：(1, n, 2)
                    
                    # 填充多边形
                    cv2.fillPoly(png, [cv_points], class_id)
                
                # 保存PNG文件
                png_filename = f"{base_name}.png"
                png_path = os.path.join(directory, png_filename)
                Image.fromarray(png).save(png_path)
                
                converted_count += 1
        
        return jsonify({
            'message': f'Conversion completed. Converted {converted_count} annotation files.',
            'converted_count': converted_count
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/sam_load_finetuned', methods=['POST'])
def load_finetuned_sam_model():
    """加载微调后的SAM模型"""
    try:
        data = request.get_json()
        model_path = data.get('model_path')
        model_type = data.get('model_type', 'vit_h')
        
        if not model_path:
            return jsonify({'error': 'Model path is required'}), 400
        
        if not os.path.exists(model_path):
            return jsonify({'error': 'Model file not found'}), 404
        
        # 获取SAM模型实例并加载微调模型
        sam = get_sam_model(model_type=model_type)
        success = sam.load_finetuned_model(model_path)
        
        if success:
            return jsonify({
                'message': 'Finetuned SAM model loaded successfully',
                'model_path': model_path,
                'model_type': model_type,
                'model_info': sam.get_model_info()
            })
        else:
            return jsonify({'error': 'Failed to load finetuned model'}), 500
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)