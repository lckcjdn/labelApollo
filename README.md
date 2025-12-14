# 图像标注工具 (Image Annotation Tool)

一个基于Web的图像标注工具，支持多边形标注、顶点编辑、自动标注等功能，适用于计算机视觉任务中的数据标注工作。

## 功能特性

- ✅ 多边形标注绘制
- ✅ 顶点编辑（添加、删除、移动）
- ✅ 标注类别管理
- ✅ 标注列表展示与管理
- ✅ 图像缩放与平移
- ✅ 自动标注功能
- ✅ 撤销操作
- ✅ 标注保存与加载

## 安装

### 1. 克隆仓库

```bash
git clone https://github.com/lckcjdn/labelApollo.git
cd labelApollo
```

### 2. 后端安装

#### 环境要求
- Python 3.7+
- pip

#### 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 前端安装

前端无需特殊安装，直接在浏览器中打开即可。

## 使用

### 1. 启动后端服务器

```bash
cd backend
python app.py
```

后端服务器将在 `http://localhost:5000` 启动。

### 2. 启动前端

#### 方法一：使用Python内置服务器

```bash
cd frontend
python -m http.server 8000
```

然后在浏览器中打开 `http://localhost:8000`

#### 方法二：直接打开HTML文件

直接在浏览器中打开 `frontend/index.html` 文件。

### 3. 基本操作

#### 标注绘制
1. 选择"绘制多边形"工具
2. 选择标注类别
3. 在图像上点击创建顶点，完成后双击闭合多边形

#### 顶点编辑
1. 选择"编辑顶点"工具
2. 点击顶点进行选择和移动
3. 按Delete键删除顶点
4. 按N键进入添加顶点模式，然后点击多边形边缘添加新顶点

#### 标注管理
1. 查看右侧的"标注列表"
2. 点击标注项可以选中并查看详细信息
3. 使用"删除"按钮删除标注

#### 图像操作
1. 鼠标滚轮缩放图像
2. alt+拖动平移图像
## 项目结构

```
image-annotation-tool/
├── backend/               # 后端代码
│   ├── app.py            # Flask应用主文件
│   ├── auto_annotate.py  # 自动标注实现
│   └── sam_model.py      # SAM模型集成
├── requirements.txt       # 项目依赖
├── frontend/             # 前端代码
│   ├── index.html        # 主页面
│   ├── script.js         # 核心功能实现
│   └── styles.css        # 样式文件
└── README.md             # 项目说明文档
```

## 技术栈

- **前端**: HTML5, CSS3, JavaScript (ES6+)
- **后端**: Python, Flask
- **图像处理**: OpenCV, Pillow
- **AI模型**: Segment Anything Model (SAM)

## 注意事项

1. 首次使用自动标注功能可能需要下载SAM模型文件
2. 建议使用Chrome或Firefox浏览器获得最佳体验
3. 大型图像可能需要较长时间加载，请耐心等待

## 许可证

MIT License

## 贡献

欢迎提交Issue和Pull Request！
