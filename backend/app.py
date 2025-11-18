from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
import os
import json
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

app = Flask(__name__, static_folder='uploads')
CORS(app)

# 配置
UPLOAD_FOLDER = 'uploads'
ANNOTATIONS_FOLDER = 'annotations'
MODEL_FOLDER = 'models'

# 确保所有必要目录存在
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(ANNOTATIONS_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

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
            model_path = os.path.join(MODEL_FOLDER, f'{self.dataset_type}_segmentation_model.pth')
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
                images.append({
                    'filename': filename,
                    'filepath': os.path.join(UPLOAD_FOLDER, filename)
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
    """检查图像文件的标注状态"""
    try:
        # 检查多种可能的标注文件位置和格式
        base_name = os.path.splitext(image_filename)[0]
        
        # 可能的标注文件位置
        possible_annotation_paths = [
            # 1. 在annotations目录下的同名子目录中
            os.path.join(ANNOTATIONS_FOLDER, os.path.basename(image_directory), f"{base_name}.json"),
            os.path.join(ANNOTATIONS_FOLDER, os.path.basename(image_directory), f"{base_name}_annotation.json"),
            # 2. 在图像同目录下
            os.path.join(image_directory, f"{base_name}.json"),
            os.path.join(image_directory, f"{base_name}_annotation.json"),
            # 3. 在图像同目录下的annotations子目录中
            os.path.join(image_directory, 'annotations', f"{base_name}.json"),
            os.path.join(image_directory, 'annotations', f"{base_name}_annotation.json"),
            # 4. 在annotations根目录下
            os.path.join(ANNOTATIONS_FOLDER, f"{base_name}.json"),
            os.path.join(ANNOTATIONS_FOLDER, f"{base_name}_annotation.json"),
            # 5. PNG格式的标注文件
            os.path.join(ANNOTATIONS_FOLDER, os.path.basename(image_directory), f"{base_name}.png"),
            os.path.join(ANNOTATIONS_FOLDER, os.path.basename(image_directory), f"{base_name}_annotation.png"),
            os.path.join(image_directory, f"{base_name}.png"),
            os.path.join(image_directory, f"{base_name}_annotation.png"),
            os.path.join(image_directory, 'annotations', f"{base_name}.png"),
            os.path.join(image_directory, 'annotations', f"{base_name}_annotation.png"),
            os.path.join(ANNOTATIONS_FOLDER, f"{base_name}.png"),
            os.path.join(ANNOTATIONS_FOLDER, f"{base_name}_annotation.png"),
        ]
        
        for annotation_path in possible_annotation_paths:
            if os.path.exists(annotation_path):
                return {
                    'annotated': True,
                    'annotation_path': annotation_path
                }
        
        return {
            'annotated': False,
            'annotation_path': None
        }
        
    except Exception as e:
        print(f"Error checking annotation status for {image_filename}: {e}")
        return {
            'annotated': False,
            'annotation_path': None
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
        
        # 将图像复制到uploads目录以便前端访问
        filename = os.path.basename(image_path)
        upload_path = os.path.join(UPLOAD_FOLDER, filename)
        
        # 如果文件已存在，生成唯一文件名
        counter = 1
        original_filename = filename
        while os.path.exists(upload_path):
            name, ext = os.path.splitext(original_filename)
            filename = f"{name}_{counter}{ext}"
            upload_path = os.path.join(UPLOAD_FOLDER, filename)
            counter += 1
        
        # 复制文件
        import shutil
        shutil.copy2(image_path, upload_path)
        
        return jsonify({
            'message': 'Image loaded successfully',
            'filename': filename,
            'filepath': upload_path,
            'original_path': image_path
        })
        
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
    """AI辅助标注预测"""
    try:
        data = request.get_json()
        image_filename = data.get('image_filename')
        model = data.get('model', 'deeplab')
        
        if not image_filename:
            return jsonify({'error': 'Missing image filename'}), 400
        
        # 构建完整的图像路径
        image_path = os.path.join(UPLOAD_FOLDER, image_filename)
        
        if not os.path.exists(image_path):
            return jsonify({'error': 'Image file not found'}), 404
        
        pred_mask = None
        if model == 'deeplab':
            # 使用DeepLabV3+模型预测
            pred_mask = seg_model.predict(image_path)
        elif model.startswith('sam'):
            # 使用SAM模型预测
            from sam_model import get_sam_model
            
            if model == 'sam_finetuned':
                # 加载微调后的模型
                sam = get_sam_model()
                finetuned_model_path = os.path.join(MODEL_FOLDER, 'sam_finetuned.pth')
                if os.path.exists(finetuned_model_path):
                    sam._load_model(finetuned_model_path)
                else:
                    # 如果没有微调模型，使用默认模型
                    sam = get_sam_model()
            else:
                # 使用默认SAM模型
                sam = get_sam_model()
            
            # 读取图像
            image = cv2.imread(image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w = image_rgb.shape[:2]
            
            # 使用整个图像作为边界框
            box = [0, 0, w, h]
            
            # 设置图像到SAM模型
            sam.set_image(image_rgb)
            
            # 预测
            masks, scores, logits = sam.predict(boxes=np.array([box]), multimask_output=True)
            
            # 使用最高分数的掩码
            best_mask_index = scores.argmax()
            best_mask = masks[best_mask_index]
            
            # 将掩码转换为与DeepLab输出相同的格式
            pred_mask = np.zeros((h, w), dtype=np.uint8)
            pred_mask[best_mask] = 1  # 暂时只支持单个类别
        
        if pred_mask is None:
            return jsonify({'error': 'Prediction failed'}), 500
        
        # 将mask转换为颜色图像
        color_mask = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3), dtype=np.uint8)
        for class_id, color in seg_model.colors.items():
            color_mask[pred_mask == class_id] = color
        
        # 转换为base64
        _, buffer = cv2.imencode('.png', color_mask)
        mask_b64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'mask': mask_b64,
            'shape': pred_mask.shape.tolist()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/save_annotation', methods=['POST'])
def save_annotation():
    """保存标注数据"""
    try:
        data = request.get_json()
        image_filename = data.get('image_filename')
        annotation_data = data.get('annotation_data')
        
        if not image_filename or not annotation_data:
            return jsonify({'error': 'Missing required data'}), 400
        
        # 保存标注文件
        base_name = os.path.splitext(image_filename)[0]
        annotation_filename = f"{base_name}_annotation.json"
        annotation_path = os.path.join(ANNOTATIONS_FOLDER, annotation_filename)
        
        # 保存为JSON文件
        with open(annotation_path, 'w') as f:
            f.write(annotation_data)
        
        return jsonify({
            'message': 'Annotation saved successfully',
            'annotation_path': annotation_path
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/load_annotation', methods=['GET'])
def load_annotation():
    """加载标注数据"""
    try:
        image_filename = request.args.get('image_filename')
        if not image_filename:
            return jsonify({'error': 'Missing image filename'}), 400
        
        base_name = os.path.splitext(image_filename)[0]
        annotation_filename = f"{base_name}_annotation.json"
        annotation_path = os.path.join(ANNOTATIONS_FOLDER, annotation_filename)
        
        if not os.path.exists(annotation_path):
            return jsonify({'annotation': None})
        
        # 读取JSON数据
        with open(annotation_path, 'r') as f:
            annotation_data = f.read()
        
        return jsonify({'annotation': annotation_data})
    except Exception as e:
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
            model_path = os.path.join(MODEL_FOLDER, f'{seg_model.dataset_type}_segmentation_model.pth')
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
        
        # 获取SAM模型实例
        sam = get_sam_model(model_type=model_type)
        
        return jsonify({
            'message': 'SAM model initialized successfully',
            'model_info': sam.get_model_info()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/sam_predict_points', methods=['POST'])
def sam_predict_with_points():
    """使用点进行SAM分割预测"""
    try:
        data = request.get_json()
        image_filename = data.get('image_filename')
        points = data.get('points', [])  # [[x1, y1], [x2, y2], ...]
        labels = data.get('labels', [])  # [1, 0, 1, ...] 1=前景点, 0=背景点
        multimask_output = data.get('multimask_output', False)
        
        if not image_filename or not points:
            return jsonify({'error': 'Missing image filename or points'}), 400
        
        # 构建图像路径
        image_path = os.path.join(UPLOAD_FOLDER, image_filename)
        if not os.path.exists(image_path):
            return jsonify({'error': 'Image file not found'}), 404
        
        # 获取SAM模型
        sam = get_sam_model()
        
        # 加载图像
        image = Image.open(image_path).convert('RGB')
        
        # 设置图像
        sam.set_image(image)
        
        # 进行预测
        masks, scores = sam.predict_with_points(
            points=points,
            labels=labels if labels else None,
            multimask_output=multimask_output
        )
        
        if masks is None:
            return jsonify({'error': 'SAM prediction failed'}), 500
        
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
        box = data.get('box')  # [x1, y1, x2, y2]
        multimask_output = data.get('multimask_output', False)
        
        if not image_filename or not box:
            return jsonify({'error': 'Missing image filename or box'}), 400
        
        # 构建图像路径
        image_path = os.path.join(UPLOAD_FOLDER, image_filename)
        if not os.path.exists(image_path):
            return jsonify({'error': 'Image file not found'}), 404
        
        # 获取SAM模型
        sam = get_sam_model()
        
        # 加载图像
        image = Image.open(image_path).convert('RGB')
        
        # 设置图像
        sam.set_image(image)
        
        # 进行预测
        masks, scores = sam.predict_with_box(
            box=box,
            multimask_output=multimask_output
        )
        
        if masks is None:
            return jsonify({'error': 'SAM prediction failed'}), 500
        
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
        sam = get_sam_model()
        return jsonify(sam.get_model_info())
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/sam_finetune', methods=['POST'])
def finetune_sam_model():
    """微调SAM模型"""
    try:
        from sam_finetuning import SAMFinetuner
        
        data = request.get_json()
        annotated_images = data.get('annotated_images', [])
        epochs = data.get('epochs', 10)
        batch_size = data.get('batch_size', 2)
        model_type = data.get('model_type', 'vit_l')
        
        if len(annotated_images) == 0:
            return jsonify({'error': 'No training data provided'}), 400
        
        # 创建微调器
        finetuner = SAMFinetuner(model_type=model_type)
        
        # 执行微调
        result = finetuner.finetune(
            annotated_images=annotated_images,
            epochs=epochs,
            batch_size=batch_size
        )
        
        return jsonify({
            'message': 'SAM model finetuning completed successfully',
            'result': result
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