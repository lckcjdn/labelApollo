# Label Apollo - AI图像标注工具

一个基于AI的图像分割标注工具，支持SAM模型微调，具备实时学习和智能辅助标注功能。

## 🌟 核心特性

### 🎯 SAM模型集成
- **SAM (Segment Anything Model)** - 最先进的图像分割模型
- **模型微调** - 支持在自定义数据集上微调SAM模型
- **多种提示方式** - 点提示、边界框提示、多模态提示
- **GPU加速** - 支持CUDA加速训练和推理

### 🎨 完整标注功能
- 支持Cityscapes和Apollo格式的图像分割标注
- 19个标准语义类别（道路、建筑、车辆、行人等）
- 可视化标注界面，支持多种标注工具
- 标注数据的保存、加载和编辑

### 🤖 AI辅助功能
- 基于SAM的智能分割
- 基于DeepLabV3+的语义分割模型
- 增量学习能力，从已标注数据中持续学习
- 智能预标注，减少人工工作量

### 📁 数据管理
- 图像文件上传和管理
- 标注数据的导入导出
- 支持外部标注文件导入
- Cityscapes和Apollo格式兼容

## 🚀 快速开始

### 方法一：一键启动（推荐）
```bash
# 双击运行启动脚本
start.bat
```

### 方法二：手动启动

1. **环境要求**
   - Python 3.8+
   - Node.js 16+
   - CUDA（可选，用于GPU加速）

2. **安装依赖**
```bash
# 后端依赖
cd backend
pip install -r requirements.txt

# 前端依赖
cd ../frontend
npm install
```

3. **下载SAM模型**
```bash
cd backend
python download_sam_model.py
```

4. **启动服务**
```bash
# 启动后端（终端1）
cd backend
python app.py

# 启动前端（终端2）
cd frontend
npm start
```

### 🛠️ 实用脚本

- **start.bat** - 一键启动所有服务
- **stop.bat** - 停止所有服务
- **test.bat** - 功能测试脚本
- **test_sam_basic.py** - SAM基础功能测试
- **test_sam_finetuning.py** - SAM微调功能测试

## 📱 访问地址

启动成功后，访问以下地址：
- **前端界面**: http://localhost:3000
- **后端API**: http://localhost:5000
- **健康检查**: http://localhost:5000/api/health

## 💡 使用指南

### SAM模型使用

1. **初始化SAM模型**
   - 在前端界面选择"SAM标注"模式
   - 系统会自动加载vit_l模型

2. **点提示标注**
   - 点击图像上的目标位置
   - 系统自动生成分割掩码

3. **边界框标注**
   - 拖拽绘制边界框
   - 系统识别框内对象

4. **模型微调**
   - 标注一定数量的样本
   - 点击"微调模型"按钮
   - 等待训练完成

### 传统标注流程

1. **上传图像**
   - 支持常见图像格式（PNG, JPG, JPEG）

2. **选择标注类别**
   - 从下拉菜单选择语义类别
   - 每个类别都有对应颜色标识

3. **开始标注**
   - 使用鼠标绘制标注区域
   - 支持多种绘制工具

4. **保存标注**
   - 自动保存到服务器
   - 支持导出标准格式

## 🧪 测试功能

### 基础功能测试
```bash
cd backend
python test_sam_basic.py
```

### 微调功能测试
```bash
cd backend
python test_sam_finetuning.py
```

### 完整系统测试
```bash
# 运行测试脚本
test.bat
```

## 📊 支持的数据格式

### Cityscapes格式
- 19个语义类别
- 标准颜色映射
- PNG掩码格式

### Apollo格式
- 道路场景专用
- 车道线检测
- 障碍物识别

## 🔧 API接口

### SAM相关
- `POST /api/sam_init` - 初始化SAM模型
- `POST /api/sam_predict_points` - 点提示预测
- `POST /api/sam_predict_box` - 边界框预测
- `POST /api/sam_finetune` - 模型微调
- `POST /api/sam_load_finetuned` - 加载微调模型

### 传统功能
- `POST /api/upload` - 上传图像
- `POST /api/save_annotation` - 保存标注
- `POST /api/predict` - AI预测
- `POST /api/train` - 模型训练

## 🎯 模型配置

### SAM模型类型
- `vit_h` - 最高精度，最大模型
- `vit_l` - 平衡精度和速度（推荐）
- `vit_b` - 最快速度，较小模型

### 微调参数
- 训练轮数：默认2轮
- 批次大小：默认1
- 学习率：自动调整

## 🐛 故障排除

### 常见问题

1. **模型初始化失败**
   - 检查模型文件是否存在
   - 验证GPU内存是否充足
   - 查看错误日志

2. **微调训练失败**
   - 确保有足够的标注数据
   - 检查CUDA是否可用
   - 验证数据格式正确

3. **预测结果异常**
   - 检查输入图像格式
   - 验证提示坐标正确
   - 重新初始化模型

### 日志查看
- 后端日志：控制台输出
- 前端日志：浏览器开发者工具
- 模型日志：训练过程详细信息

## 🏗️ 项目结构

```
labelApollo/
├── backend/                 # 后端代码
│   ├── app.py              # Flask应用主文件
│   ├── sam_model.py        # SAM模型封装
│   ├── sam_finetuning.py   # 微调功能
│   ├── models/             # 模型文件目录
│   ├── uploads/            # 上传文件目录
│   ├── annotations/        # 标注文件目录
│   └── test_*.py           # 测试脚本
├── frontend/               # 前端代码
│   ├── src/                # React源码
│   ├── public/             # 静态资源
│   └── package.json        # 前端依赖
├── start.bat               # 启动脚本
├── stop.bat                # 停止脚本
├── test.bat                # 测试脚本
└── README.md               # 项目文档
```

## 🤝 贡献指南

1. Fork项目
2. 创建功能分支
3. 提交更改
4. 推送到分支
5. 创建Pull Request

## 📄 许可证

本项目采用MIT许可证 - 查看 [LICENSE](LICENSE) 文件了解详情

## 🙏 致谢

- [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything) - Meta AI的分割模型
- [Segmentation Models PyTorch](https://github.com/qubvel/segmentation_models.pytorch) - 语义分割模型库
- [Cityscapes Dataset](https://www.cityscapes-dataset.com/) - 城市场景数据集

---

🚀 **开始您的AI标注之旅！**