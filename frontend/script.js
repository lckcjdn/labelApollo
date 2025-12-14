// 图像标注工具核心功能
class ImageAnnotationTool {
    constructor() {
        this.currentImageIndex = -1;
        this.images = [];
        this.annotations = {};
        this.currentTool = 'draw-tool';
        this.currentCategory = 'car';
        this.isDrawing = false;
        this.currentPolygon = [];
        this.selectedPolygon = null;
        this.selectedVertex = null;
        this.imageScale = 1;
        // 图像缩放和平移
        this.zoom = 1.0;
        this.translateX = 0;
        this.translateY = 0;
        this.isPanning = false;
        this.isMovingVertex = false;
        this.panStartX = 0;
        this.panStartY = 0;
        
        // 撤销功能相关
        this.undoHistory = {}; // 存储历史状态的对象，键为图像名称，值为该图像的历史记录数组
        this.undoHistoryIndex = {}; // 当前历史记录索引对象，键为图像名称，值为该图像当前的历史记录索引
        this.maxHistorySize = 50; // 最大历史记录数量限制
        
        // 获取DOM元素
        this.canvas = document.getElementById('annotation-canvas');
        this.ctx = this.canvas.getContext('2d');
        this.fileList = document.getElementById('file-list');
        this.categorySelect = document.getElementById('category-select');
        this.toolButtons = document.querySelectorAll('.tool-btn');
        this.openFolderBtn = document.getElementById('open-folder-btn');
        this.saveBtn = document.getElementById('save-btn');
        this.autoAnnotateBtn = document.getElementById('auto-annotate-btn');
        this.modelSelect = document.getElementById('model-select');
        this.confidenceSlider = document.getElementById('confidence-slider');
        this.confidenceValue = document.getElementById('confidence-value');
        this.prevBtn = document.getElementById('prev-btn');
        this.nextBtn = document.getElementById('next-btn');
        this.addCategoryBtn = document.getElementById('add-category-btn');
        this.importCategoryBtn = document.getElementById('import-category-btn');
        this.importCategoriesFile = document.getElementById('import-categories-file');
        // 初始化配置文件输入元素
        this.configFileInput = document.createElement('input');
        this.configFileInput.type = 'file';
        this.configFileInput.accept = '.yaml,.yml,.json';
        this.configFileInput.onchange = (e) => this.handleConfigFile(e);
        this.fileInput = document.createElement('input');
        this.fileInput.type = 'file';
        this.fileInput.accept = 'image/*';
        this.fileInput.multiple = true;
        this.fileInput.webkitdirectory = true;
        this.fileInput.onchange = (e) => this.handleFileSelect(e);
        
        // 初始化类别颜色映射
        this.categoryColors = {
            'car': '#ff0000',
            'person': '#00ff00',
            'bike': '#0000ff',
            'traffic_light': '#ffff00',
            'road': '#808080',
            'building': '#c0c0c0'
        };

        // 自动保存状态
        this.hasUnsavedChanges = false;
        this.lastSaveTime = Date.now();
        this.autoSaveInterval = 30000; // 30秒自动保存
        this.autoSaveTimer = null; // 初始化自动保存计时器
        
        // 添加新顶点模式
        this.isAddingVertex = false;

        // 初始化事件监听
        this.initEventListeners();
        this.initAutoSave();
        // 初始化鼠标指针样式
        this.updateCursorStyle();
    }
    
    // 初始化事件监听
    initEventListeners() {
        // 工具栏按钮事件
        this.openFolderBtn.addEventListener('click', () => this.fileInput.click());
        this.saveBtn.addEventListener('click', () => this.saveAnnotations());
        this.autoAnnotateBtn.addEventListener('click', () => this.autoAnnotate());
        this.prevBtn.addEventListener('click', () => this.showPreviousImage());
        this.nextBtn.addEventListener('click', () => this.showNextImage());
        
        // 置信度滑块事件
        if (this.confidenceSlider) {
            this.confidenceSlider.addEventListener('input', (e) => {
                this.confidenceValue.textContent = e.target.value;
            });
        }
        
        // 工具按钮事件
        this.toolButtons.forEach(btn => {
            btn.addEventListener('click', (e) => this.selectTool(e));
        });
        
        // 设置画布事件监听
        this.canvas.addEventListener('mousedown', (e) => this.handleCanvasMouseDown(e));
        this.canvas.addEventListener('mousemove', (e) => this.handleCanvasMouseMove(e));
        this.canvas.addEventListener('mouseup', (e) => this.handleCanvasMouseUp(e));
        this.canvas.addEventListener('mouseleave', (e) => this.handleCanvasMouseLeave(e));
        this.canvas.addEventListener('wheel', (e) => this.handleCanvasWheel(e));
        this.canvas.addEventListener('dblclick', (e) => this.handleCanvasDblClick(e));
        this.canvas.addEventListener('contextmenu', (e) => this.handleCanvasContextMenu(e)); // 右键菜单事件
        
        // 键盘事件
        document.addEventListener('keydown', (e) => this.handleKeyDown(e));
        
        // N键添加新顶点
        document.addEventListener('keydown', (e) => {
            if (e.key.toLowerCase() === 'n' && this.currentTool === 'edit-tool') {
                e.preventDefault();
                this.isAddingVertex = !this.isAddingVertex;
                // 更新鼠标指针样式
                this.updateCursorStyle();
            }
        });
        
        // Delete键删除选中的顶点
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Delete' && this.selectedVertex && this.selectedPolygon) {
                // 确保删除后多边形至少有3个点（保持多边形有效性）
                if (this.selectedPolygon.points.length > 3) {
                    const vertexIndex = this.selectedPolygon.points.indexOf(this.selectedVertex);
                    this.selectedPolygon.points.splice(vertexIndex, 1);
                    this.selectedVertex = null;
                    this.redraw();
                    this.setUnsavedChanges();
                } else {
                    alert('多边形至少需要3个顶点，无法删除！');
                }
            }
        });
        
        // 类别选择事件
        // 为类别列表项添加点击事件
        this.categorySelect.addEventListener('click', (e) => {
            if (e.target.classList.contains('category-item')) {
                // 移除所有项的active类
                [...this.categorySelect.children].forEach(item => {
                    item.classList.remove('active');
                });
                // 为当前点击项添加active类
                e.target.classList.add('active');
                // 更新当前类别
                this.currentCategory = e.target.dataset.value;
            }
        });
        
        // 自定义标签事件
        this.addCategoryBtn.addEventListener('click', () => this.addCategory());
        this.importCategoryBtn.addEventListener('click', () => this.importCategoriesFile.click());
        this.importCategoriesFile.addEventListener('change', (e) => this.handleImportCategories(e));
        // 添加加载Apollo配置文件的事件监听
        this.loadConfigBtn = document.getElementById('load-config-btn');
        if (this.loadConfigBtn) {
            this.loadConfigBtn.addEventListener('click', () => this.loadConfigFile());
        }
        
        // 窗口大小变化事件
        window.addEventListener('resize', () => this.resizeCanvas());
    }
    
    // 处理文件选择（包括图像和JSON标注）
    handleFileSelect(event) {
        const files = Array.from(event.target.files);
        
        // 分离图像和标注文件
        const imageFiles = [];
        const jsonFiles = files.filter(file => file.name.endsWith('.json'));
        const pngFiles = files.filter(file => file.name.endsWith('.png'));
        
        // 检查同名PNG是否为标注文件（仍然需要为非PNG图像寻找对应的标注PNG）
        const imageNameToPNGMap = {};
        
        // 检查同名PNG是否为标注文件
        pngFiles.forEach(pngFile => {
            // 检查是否存在同名的非PNG图像文件
            const baseName = pngFile.name.replace('.png', '');
            
            // 尝试找到同名的非PNG图像文件
            for (const file of files) {
                if (file !== pngFile && 
                    file.name.startsWith(baseName) && 
                    file.name !== pngFile.name && 
                    file.type.startsWith('image/')) {
                    // 找到了同名的非PNG图像，这个PNG可能是标注文件
                    imageNameToPNGMap[file.name] = pngFile;
                    break;
                }
            }
        });
        
        // 只添加非PNG图像文件（不加载PNG图像）
        const nonPngImageFiles = files.filter(file => 
            file.type.startsWith('image/') && !file.name.endsWith('.png')
        );
        imageFiles.push(...nonPngImageFiles);
        
        // 尝试获取文件的完整路径（受浏览器安全限制）
        this.images = imageFiles.map(file => {
            // 创建一个新的对象，包含原始文件和路径信息
            const imageObj = {
                name: file.name,
                type: file.type,
                size: file.size,
                lastModified: file.lastModified,
                imageObject: null,
                filePath: file.webkitRelativePath || file.name, // 尝试获取相对路径或使用文件名
                file: file,
                // 如果有对应的PNG标注文件，添加到图像对象中
                annotationPNG: imageNameToPNGMap[file.name] || null
            };
            
            return imageObj;
        });
        
        this.loadImages();
        this.updateFileList();
        this.updateStats();
        
        // 加载对应的JSON标注
        if (jsonFiles.length > 0) {
            this.loadLabelMeAnnotations(jsonFiles);
        }
        
        // 加载对应的PNG标注
        if (pngFiles.length > 0) {
            this.loadPNGAnnotation(pngFiles);
        }
        
        if (imageFiles.length > 0) {
            this.currentImageIndex = 0;
            this.loadImage(this.images[0]);
        }
    };
    
    // 自动查找并加载对应图像的标注文件
    autoLoadAnnotationForImage(imageName) {
        // 从后端加载标注文件，不再自动弹出文件选择对话框
        
        // 获取当前图像的完整路径信息
        let imageFilepath = '';
        let imageDirectory = '';
        
        // 查找当前图像对象
        const currentImage = this.images.find(img => img.name === imageName);
        if (currentImage && currentImage.filePath) {
            imageFilepath = currentImage.filePath;
            // 从filePath中提取目录，确保正确处理路径分隔符
            const lastSlashIndex = Math.max(
                imageFilepath.lastIndexOf('/'),
                imageFilepath.lastIndexOf('\\')
            );
            if (lastSlashIndex !== -1) {
                imageDirectory = imageFilepath.substring(0, lastSlashIndex + 1);
            }
        }
        
        // 构建请求参数
        const params = new URLSearchParams();
        params.append('image_filename', imageName);
        if (imageFilepath) {
            params.append('image_filepath', imageFilepath);
        }
        if (imageDirectory) {
            params.append('output_folder', imageDirectory);
        }
        
        // 调用后端API尝试加载标注文件
        fetch(`http://localhost:5000/api/load_annotation?${params.toString()}`)
            .then(response => response.json())
            .then(data => {
                if (!data.error && data.annotation) {
                    // 利用后端返回的format字段来确定文件格式
                    if (data.format === 'json') {
                        try {
                            const jsonData = JSON.parse(data.annotation);
                            this.processLoadedAnnotations(imageName, jsonData, 'json');
                        } catch (e) {
                            // 静默失败，不显示错误提示
                        }
                    } else if (data.format === 'png') {
                        this.processLoadedAnnotations(imageName, data.annotation, 'png');
                    } else {
                        // 兼容旧版本，尝试自动解析
                        try {
                            // 尝试解析为JSON
                            const jsonData = JSON.parse(data.annotation);
                            this.processLoadedAnnotations(imageName, jsonData, 'json');
                        } catch (e) {
                            // 不是JSON，可能是base64编码的PNG
                            this.processLoadedAnnotations(imageName, data.annotation, 'png');
                        }
                    }
                }
                // 未找到标注文件时不显示提示
            })
            .catch(error => {
                // 静默失败，不显示错误提示
            });
        
        // 更新文件列表以显示标注状态
        this.updateFileList();
    };
    
    // 处理从后端加载的标注数据
    processLoadedAnnotations(imageName, data, type) {
        try {
            if (type === 'json') {
                // 处理JSON格式
                const labelmeJSON = data;
                
                // 处理标注数据 - 确保shapes字段存在且是数组
                const shapes = Array.isArray(labelmeJSON.shapes) ? labelmeJSON.shapes : [];
                
                // 创建标注数据
                const annotations = shapes.map((shape) => ({
                    id: Date.now() + Math.random() * 1000,
                    classId: shape.label || 'unknown',
                    points: Array.isArray(shape.points) ? shape.points : [],
                    timestamp: new Date().toISOString()
                })).filter(annotation => annotation.points.length >= 3);
                
                // 保存标注
                this.annotations[imageName] = annotations;
                
                // 更新标注列表
                this.updateAnnotationsList();
                
                // 立即重绘以显示标注
                setTimeout(() => {
                    this.redraw();
                }, 100);
            } else if (type === 'png') {
                // 处理PNG格式标注
                const image = new Image();
                image.onload = () => {
                    // 创建临时画布处理PNG标注
                    const tempCanvas = document.createElement('canvas');
                    tempCanvas.width = image.width;
                    tempCanvas.height = image.height;
                    const tempCtx = tempCanvas.getContext('2d');
                    tempCtx.drawImage(image, 0, 0);
                    
                    // 获取图像数据
                    const imageData = tempCtx.getImageData(0, 0, image.width, image.height);
                    const data = imageData.data;
                    
                    // 从PNG mask中提取标注信息
                    // 注：这里简化处理，实际应用中需要实现更复杂的多边形提取算法
                    const annotations = [];
                    
                    // 遍历图像数据，查找非零区域（这里是一个简化实现）
                    // 实际应该使用轮廓提取算法来生成多边形
                    // 由于缺少完整的多边形提取逻辑，我们创建一个简单的标记
                    this.annotations[imageName] = [{ 
                        id: Date.now() + Math.random() * 1000,
                        classId: 'from_png_mask',
                        mask_data: data, // 保存原始图像数据数组
                        points: [], // 实际应用中应该填充从mask中提取的多边形点
                        timestamp: new Date().toISOString()
                    }];
                    
                    console.log(`保存了PNG掩码数据，需要实现多边形提取逻辑`);
                    
                    console.log(`Loaded PNG annotation for ${imageName}`);
                    
                    // 更新界面
                    this.updateFileList();
                    this.updateStats();
                    this.redraw();
                };
                image.src = `data:image/png;base64,${data}`;
            }
            
            // 更新界面
            this.updateFileList();
            this.updateStats();
            this.redraw();
            
        } catch (error) {
            console.error('处理加载的标注文件失败:', error);
        }
    }
    
    // 加载标注文件（支持JSON和PNG格式）
    loadAnnotations(files) {
        files.forEach(file => {
            // 根据文件扩展名判断格式
            if (file.name.endsWith('.json')) {
                this.loadLabelMeAnnotations([file]);
            } else if (file.name.endsWith('.png')) {
                this.loadPNGAnnotation([file]);
            } else {
                alert(`不支持的文件格式: ${file.name}`);
            }
        });
    }
    
    // 加载LabelMe格式的标注
    loadLabelMeAnnotations(jsonFiles) {
        jsonFiles.forEach(file => {
            const reader = new FileReader();
            reader.onload = (event) => {
                try {
                    const labelmeJSON = JSON.parse(event.target.result);
                    
                    // 确定对应的图像名称 - 增强健壮性
                    let imageName = '';
                    let matchingImage = null;
                    
                    // 方法1: 尝试从JSON的imagePath字段获取
                    if (labelmeJSON.imagePath) {
                        try {
                            imageName = labelmeJSON.imagePath.split(/[/\\]/).pop();
                            console.log(`从imagePath获取图像名称: ${imageName}`);
                            // 检查是否有匹配的图像
                            matchingImage = this.images.find(img => img.name === imageName);
                        } catch (e) {
                            console.warn('从imagePath解析图像名称失败:', e);
                        }
                    }
                    
                    // 方法2: 如果imagePath不可用或解析失败，或找不到匹配的图像，从标注文件名推断
                    if (!matchingImage && file.name) {
                        // 从标注文件名(如image.json)推断图像名(image.jpg/png等)
                        const baseName = file.name.replace(/\.json$/i, '');
                        // 尝试匹配当前加载的图像
                        matchingImage = this.images.find(img => {
                            const imgBaseName = img.name.replace(/\.[^/.]+$/, '');
                            return imgBaseName === baseName;
                        });
                        
                        if (matchingImage) {
                            imageName = matchingImage.name;
                            console.log(`从标注文件名和当前图像列表匹配获取图像名称: ${imageName}`);
                        }
                    }
                    
                    // 如果找到匹配的图像，处理标注
                    if (matchingImage) {
                        // 处理标注数据 - 确保shapes字段存在且是数组
                        const shapes = Array.isArray(labelmeJSON.shapes) ? labelmeJSON.shapes : [];
                        
                        // 创建标注数据
                        const annotations = shapes.map(shape => ({
                            id: Date.now() + Math.random() * 1000,
                            classId: shape.label || 'unknown', // 提供默认类别
                            points: Array.isArray(shape.points) ? shape.points : [], // 确保points是数组
                            timestamp: new Date().toISOString()
                        })).filter(annotation => annotation.points.length >= 3); // 只保留有效的多边形(至少3个点)
                        
                        // 保存标注
                        this.annotations[imageName] = annotations;
                        
                        // 更新界面
                        this.updateFileList();
                        this.updateStats();
                        
                        // 如果当前显示的是这个图像，重新绘制
                        if (this.images[this.currentImageIndex].name === imageName) {
                            this.redraw();
                        }
                        
                        console.log(`Loaded annotations for ${imageName}: ${annotations.length} shapes`);
                    } else {
                        // 找不到匹配的图像，记录警告但不报错
                        console.warn(`无法为标注文件 ${file.name} 找到匹配的图像`);
                    }
                    
                } catch (error) {
                    console.error('加载标注文件失败:', error);
                    // 更友好的错误提示
                    // alert(`加载标注文件失败: ${error.message || '未知错误'}\n请检查文件格式是否正确`);
                }
            };
            reader.readAsText(file);
        });
    }
    
    // 加载PNG格式的标注
    loadPNGAnnotation(pngFiles) {
        pngFiles.forEach(file => {
            const reader = new FileReader();
            reader.onload = (event) => {
                try {
                    // 确定对应的图像名称
                    let imageName = '';
                    let matchingImage = null;
                    
                    // 从PNG文件名推断图像名
                    if (file.name) {
                        const baseName = file.name.replace(/\.png$/i, '');
                        // 尝试匹配当前加载的图像
                        matchingImage = this.images.find(img => {
                            const imgBaseName = img.name.replace(/\.[^/.]+$/, '');
                            return imgBaseName === baseName;
                        });
                        
                        if (matchingImage) {
                            imageName = matchingImage.name;
                            console.log(`从PNG文件名匹配获取图像名称: ${imageName}`);
                        }
                    }
                    
                    // 如果找到匹配的图像，处理标注
                    if (matchingImage) {
                        // 直接保存PNG数据
                        const pngData = event.target.result.split(',')[1]; // 获取base64数据
                        
                        // 这里只是简单保存PNG数据，实际应用中可能需要解析mask并转换为多边形
                        this.annotations[imageName] = [{ 
                            id: Date.now() + Math.random() * 1000,
                        classId: 'from_png_mask',
                        mask: pngData,
                        timestamp: new Date().toISOString()
                    }];
                    
                    // 更新界面
                    this.updateFileList();
                    this.updateStats();
                    this.redraw();
                    
                    console.log(`Loaded PNG annotation for ${imageName}`);
                    alert(`成功加载PNG标注文件`);
                    } else {
                        console.warn(`没有找到与PNG标注文件 ${file.name} 匹配的图像`);
                    }
                    
                } catch (error) {
                    console.error('加载PNG标注文件失败:', error);
                    alert(`加载PNG标注文件失败: ${error.message || '未知错误'}`);
                }
            };
            reader.readAsDataURL(file); // 读取为DataURL以获取base64数据
        });
    }
    
    // 加载图像
    loadImages() {
        this.images.forEach((image, index) => {
            const img = new Image();
            // 使用原始文件创建URL
            const fileObj = image.file;
            img.src = URL.createObjectURL(fileObj);
            image.imageObject = img;
        });
    }
    
    // 加载当前图像到画布
    loadImage(imageFile) {
        const img = imageFile.imageObject;
        
        // 定义图片加载完成后的处理函数
        const handleImageLoad = () => {
            // 设置画布大小
            this.canvas.width = img.width;
            this.canvas.height = img.height;
            
            // 绘制图像
            this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
            this.ctx.drawImage(img, 0, 0, this.canvas.width, this.canvas.height);
            
            // 绘制标注
            this.drawAnnotations(imageFile.name);
            
            // 更新状态栏
            this.updateStatusBar(imageFile.name);
            
            // 为当前图像初始化历史记录（如果不存在）
            if (!this.undoHistory[imageFile.name]) {
                this.undoHistory[imageFile.name] = [];
            }
            if (this.undoHistoryIndex[imageFile.name] === undefined) {
                this.undoHistoryIndex[imageFile.name] = -1;
            }
            
            // 如果历史记录为空，保存当前状态作为初始状态
            if (this.undoHistory[imageFile.name].length === 0) {
                this.saveStateToHistory();
            }
            
            // 首先检查是否有前端直接加载的PNG标注文件
            if (imageFile.annotationPNG) {
                console.log(`发现同名PNG标注文件，直接加载: ${imageFile.annotationPNG.name}`);
                const reader = new FileReader();
                reader.onload = (event) => {
                    // 获取base64编码的图像数据
                    const base64Data = event.target.result;
                    // 处理PNG标注
                    this.processLoadedAnnotations(imageFile.name, base64Data, 'png');
                };
                reader.readAsDataURL(imageFile.annotationPNG);
            } else {
                // 自动从后端加载标注文件，但不再弹出文件选择对话框
                this.autoLoadAnnotationForImage(imageFile.name);
            }
        };
        
        // 设置图片加载事件
        img.onload = handleImageLoad.bind(this);
        
        // 如果图片已经加载完成，直接调用处理函数
        if (img.complete && img.naturalWidth > 0) {
            handleImageLoad.call(this);
        }
    }
    
    // 更新文件列表
    updateFileList() {
        this.fileList.innerHTML = '';
        
        this.images.forEach((image, index) => {
            const li = document.createElement('li');
            li.textContent = image.name;
            li.dataset.index = index;
            
            // 标记已标注和未标注
            if (this.annotations[image.name] && this.annotations[image.name].length > 0) {
                li.classList.add('annotated');
            } else {
                li.classList.add('unannotated');
            }
            
            // 设置当前图像的激活状态
            if (index === this.currentImageIndex) {
                li.classList.add('active');
            }
            
            // 添加点击事件
            li.addEventListener('click', () => {
                this.currentImageIndex = index;
                this.loadImage(image);
                this.updateFileList();
            });
            
            this.fileList.appendChild(li);
        });
    }
    
    // 处理画布鼠标按下事件
    handleCanvasMouseDown(e) {
        const rect = this.canvas.getBoundingClientRect();
        // 计算画布渲染尺寸与内部像素尺寸的比例
        const scaleX = this.canvas.width / rect.width;
        const scaleY = this.canvas.height / rect.height;
        // 获取缩放后的真实画布坐标
        const canvasX = (e.clientX - rect.left) * scaleX;
        const canvasY = (e.clientY - rect.top) * scaleY;
        
        // 检查是否在平移模式
        if (this.currentTool === 'pan-tool' || e.ctrlKey || e.altKey || e.shiftKey) {
            // 进入平移模式
            this.isPanning = true;
            this.panStartX = canvasX;
            this.panStartY = canvasY;
            this.canvas.style.cursor = 'grab';
        } else {
            const x = (canvasX - this.translateX) / this.zoom;
            const y = (canvasY - this.translateY) / this.zoom;

            if (this.currentTool === 'draw-tool') {
                // 如果还没开始绘制，则开始绘制
                if (!this.isDrawing) {
                    this.startDrawing(x, y);
                }
            } else if (this.currentTool === 'edit-tool') {
                // 按下N键时，尝试在边缘添加顶点
                if (this.isAddingVertex) {
                    const { polygon, edgeIndex } = this.findNearestEdge(x, y);
                    if (polygon && edgeIndex !== -1) {
                        // 保存当前状态到历史记录
                        this.saveStateToHistory();
                        
                        // 在边缘上插入新顶点
                        polygon.points.splice(edgeIndex, 0, [x, y]);
                        
                        // 选择新添加的顶点
                        this.selectedPolygon = polygon;
                        this.selectedVertex = polygon.points[edgeIndex];
                        
                        this.redraw();
                        this.setUnsavedChanges();
                        this.isAddingVertex = false;
                        // 恢复鼠标指针样式
                        this.canvas.style.cursor = 'default';
                        return;
                    }
                }
                
                this.selectVertex(x, y);
                // 设置移动顶点状态
                this.isMovingVertex = this.selectedVertex !== null;
                if (this.isMovingVertex) {
                    this.canvas.style.cursor = 'move';
                }
            } else if (this.currentTool === 'delete-tool') {
                this.deleteAnnotation(x, y);
            }
        }
        
        e.preventDefault();
    }

    // 删除标注
    deleteAnnotation(x, y, tolerance = 20) {
        const currentImageName = this.images[this.currentImageIndex].name;
        const imageAnnotations = this.annotations[currentImageName] || [];
        const annotationsToRemove = [];

        // 检查点是否在多边形内部
        for (let i = 0; i < imageAnnotations.length; i++) {
            const polygon = imageAnnotations[i];
            if (this.isPointInPolygon({x, y}, polygon.points)) {
                annotationsToRemove.push(i);
            }
        }

        // 如果有标注要删除，则保存当前状态
        if (annotationsToRemove.length > 0) {
            this.saveStateToHistory();
            
            // 移除标注
            for (let i = annotationsToRemove.length - 1; i >= 0; i--) {
                imageAnnotations.splice(annotationsToRemove[i], 1);
            }

            this.redraw();
            this.updateFileList();
            this.updateStats();
            this.setUnsavedChanges();
        }
    }

    // 检查点是否在多边形内部 (射线法)
    isPointInPolygon(point, polygon) {
        let inside = false;
        for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
            const xi = polygon[i][0], yi = polygon[i][1];
            const xj = polygon[j][0], yj = polygon[j][1];

            const intersect = ((yi > point.y) !== (yj > point.y)) && 
                (point.x < (xj - xi) * (point.y - yi) / (yj - yi) + xi);
            if (intersect) inside = !inside;
        }
        return inside;
    }
    
    // 开始绘制多边形
    startDrawing(x, y) {
        this.isDrawing = true;
        this.currentPolygon = [[x, y]];
        this.selectedPolygon = null;
        this.selectedVertex = null;
        
        // 绘制第一个点 (应用缩放和平移变换)
        this.drawPoint(x, y, '#ff0000');
    }
    
    // 处理画布鼠标移动事件
    handleCanvasMouseMove(e) {
        const rect = this.canvas.getBoundingClientRect();
        // 计算画布渲染尺寸与内部像素尺寸的比例
        const scaleX = this.canvas.width / rect.width;
        const scaleY = this.canvas.height / rect.height;
        // 获取缩放后的真实画布坐标
        const canvasX = (e.clientX - rect.left) * scaleX;
        const canvasY = (e.clientY - rect.top) * scaleY;
        let x = (canvasX - this.translateX) / this.zoom;
        let y = (canvasY - this.translateY) / this.zoom;
        
        // 更改鼠标指针样式
        if (!this.isPanning && !this.isDrawing && !this.isMovingVertex && this.currentTool === 'edit-tool') {
            // 如果在添加顶点模式下
            if (this.isAddingVertex) {
                // 检查是否悬停在边缘上
                const { edge } = this.findNearestEdge(x, y);
                this.canvas.style.cursor = edge !== null ? 'crosshair' : 'not-allowed';
            } else {
                // 临时保存当前选中状态
                const tempSelectedPolygon = this.selectedPolygon;
                const tempSelectedVertex = this.selectedVertex;
                
                // 检查是否悬停在顶点上
                this.selectVertex(x, y);
                const isOverVertex = this.selectedVertex !== null;
                
                // 恢复之前的选中状态
                this.selectedPolygon = tempSelectedPolygon;
                this.selectedVertex = tempSelectedVertex;
                
                // 检查是否悬停在边缘上
                let isOverEdge = false;
                if (!isOverVertex) {
                    const { edge } = this.findNearestEdge(x, y);
                    isOverEdge = edge !== null;
                }
                
                // 设置鼠标指针样式
                if (isOverVertex) {
                    this.canvas.style.cursor = 'move';
                } else if (isOverEdge) {
                    this.canvas.style.cursor = 'crosshair';
                } else {
                    this.canvas.style.cursor = 'default';
                }
            }
        }
        
        if (this.isPanning) {
            // 平移画布
            const deltaX = canvasX - this.panStartX;
            const deltaY = canvasY - this.panStartY;
            this.translateX += deltaX;
            this.translateY += deltaY;
            this.panStartX = canvasX;
            this.panStartY = canvasY;
            this.redraw();
        } else if (this.isDrawing) {
            // 重绘图像和当前多边形
            this.redraw();
            
            // 绘制当前多边形的所有线段
            if (this.currentPolygon.length > 1) {
                this.ctx.beginPath();
                this.ctx.moveTo((this.currentPolygon[0][0] * this.zoom) + this.translateX, (this.currentPolygon[0][1] * this.zoom) + this.translateY);
                for (let i = 1; i < this.currentPolygon.length; i++) {
                    this.ctx.lineTo((this.currentPolygon[i][0] * this.zoom) + this.translateX, (this.currentPolygon[i][1] * this.zoom) + this.translateY);
                }
                // 绘制当前鼠标位置与最后一个点之间的连线
                this.ctx.lineTo((x * this.zoom) + this.translateX, (y * this.zoom) + this.translateY);
                this.ctx.strokeStyle = '#ff0000';
                this.ctx.lineWidth = 2 * this.zoom;
                this.ctx.stroke();
            }
            
            // 绘制当前鼠标位置 (使用图像坐标)
            this.drawPoint(x, y, '#00ff00');
        } else if (this.isMovingVertex && this.selectedVertex) {
            // 移动选中的顶点
            this.selectedVertex[0] = x;
            this.selectedVertex[1] = y;
            this.redraw();
            // 标记为未保存状态
            this.setUnsavedChanges();
        }
    }
    
    // 处理画布鼠标释放事件
    handleCanvasMouseUp(e) {
        const rect = this.canvas.getBoundingClientRect();
        // 计算画布渲染尺寸与内部像素尺寸的比例
        const scaleX = this.canvas.width / rect.width;
        const scaleY = this.canvas.height / rect.height;
        // 获取缩放后的真实画布坐标
        const canvasX = (e.clientX - rect.left) * scaleX;
        const canvasY = (e.clientY - rect.top) * scaleY;
        const x = (canvasX - this.translateX) / this.zoom;
        const y = (canvasY - this.translateY) / this.zoom;
        
        if (this.isPanning) {
            // 退出平移模式
            this.isPanning = false;
            this.canvas.style.cursor = 'default';
        } else if (this.isDrawing) {
            // 检查是否闭合多边形
            if (this.currentPolygon.length > 2 && this.isCloseToFirstPoint(x, y)) {
                // 闭合多边形
                this.savePolygon();
                this.isDrawing = false;
                this.currentPolygon = [];
            } else {
                // 添加新点
                this.currentPolygon.push([x, y]);
            }
            
            this.redraw();
        } else if (this.isMovingVertex && this.selectedVertex) {
            // 在移动顶点之前保存当前状态
            if (this.selectedVertex) {
                this.saveStateToHistory();
            }
            
            // 移动顶点结束
            this.selectedVertex[0] = x;
            this.selectedVertex[1] = y;
            this.isMovingVertex = false;
            this.redraw();
            this.setUnsavedChanges();
            // 保持顶点选中状态，让用户可以继续编辑其他顶点
        }
    }
    
    // 处理画布鼠标离开事件
    handleCanvasMouseLeave(e) {
        if (this.isDrawing) {
            this.isDrawing = false;
            this.currentPolygon = [];
            this.redraw();
        }
        this.isPanning = false;
        this.isMovingVertex = false;
        this.canvas.style.cursor = 'default';
    }

    // 处理画布滚轮事件（缩放）
    handleCanvasWheel(e) {
        e.preventDefault();
        const rect = this.canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        
        // 计算缩放因子
        const zoomSpeed = 0.1;
        const delta = e.deltaY > 0 ? -zoomSpeed : zoomSpeed;
        const newZoom = Math.max(0.1, Math.min(10, this.zoom + delta));
        
        // 调整平移量以保持鼠标位置不变
        this.translateX = x - (x - this.translateX) * (newZoom / this.zoom);
        this.translateY = y - (y - this.translateY) * (newZoom / this.zoom);
        
        this.zoom = newZoom;
        this.redraw();
    }

    // 处理画布双击事件（重置缩放）
    handleCanvasDblClick(e) {
        this.zoom = 1.0;
        this.translateX = 0;
        this.translateY = 0;
        this.redraw();
    }
    
    // 处理键盘事件
    handleKeyDown(e) {
        // 检查是否有输入框或文本区域获得焦点
        const activeElement = document.activeElement;
        if (activeElement && (activeElement.tagName === 'INPUT' || activeElement.tagName === 'TEXTAREA')) {
            return;
        }

        // 处理Ctrl+S快捷键保存功能
        if (e.ctrlKey && e.key === 's') {
            e.preventDefault();
            this.saveAnnotations();
            return;
        }
        
        // 处理Ctrl+Z快捷键撤销功能
        if (e.ctrlKey && e.key === 'z') {
            e.preventDefault();
            this.undo();
            return;
        }
        
        switch (e.key.toLowerCase()) {
            case 'p':
                // 切换到平移工具
                this.switchTool('pan-tool');
                break;
            case 'd':
                // 切换到绘制工具
                this.switchTool('draw-tool');
                break;
            case 'e':
                // 切换到编辑工具
                this.switchTool('edit-tool');
                break;
            case 'delete':
            case 'backspace':
                // 删除标注或顶点
                if (this.currentTool === 'edit-tool') {
                    this.deleteSelectedVertex();
                } else {
                    this.deleteSelectedAnnotation();
                }
                break;
            case 'escape':
                // 取消绘制
                if (this.isDrawing) {
                    this.isDrawing = false;
                    this.currentPolygon = [];
                    this.redraw();
                }
                break;
            case 'space':
                // 切换到平移工具（替代之前的空格快捷键）
                if (this.currentTool !== 'pan-tool') {
                    this.switchTool('pan-tool');
                } else {
                    this.switchTool('draw-tool');
                }
                break;
            case '1':
                // 切换到绘制工具
                this.switchTool('draw-tool');
                break;
            case '2':
                // 切换到编辑工具
                this.switchTool('edit-tool');
                break;
            case '3':
                // 切换到删除工具
                this.switchTool('delete-tool');
                break;
            case '4':
                // 切换到平移工具
                this.switchTool('pan-tool');
                break;
        }
    }
    
    // 删除选中的顶点
    deleteSelectedVertex() {
        if (!this.selectedPolygon || !this.selectedVertex) return;
        
        // 在删除顶点之前保存当前状态到历史记录
        this.saveStateToHistory();
        
        const currentImageName = this.images[this.currentImageIndex].name;
        const imageAnnotations = this.annotations[currentImageName] || [];
        
        // 找到顶点在多边形中的索引
        const vertexIndex = this.selectedPolygon.points.indexOf(this.selectedVertex);
        if (vertexIndex === -1) return;
        
        // 删除顶点
        this.selectedPolygon.points.splice(vertexIndex, 1);
        
        // 如果多边形顶点少于3个，删除整个多边形
        if (this.selectedPolygon.points.length < 3) {
            const polygonIndex = imageAnnotations.indexOf(this.selectedPolygon);
            if (polygonIndex !== -1) {
                imageAnnotations.splice(polygonIndex, 1);
            }
            this.selectedPolygon = null;
            this.selectedVertex = null;
        } else {
            // 选择下一个顶点（如果删除的是最后一个顶点，则选择第一个顶点）
            const newVertexIndex = Math.min(vertexIndex, this.selectedPolygon.points.length - 1);
            this.selectedVertex = this.selectedPolygon.points[newVertexIndex];
        }
        
        this.redraw();
        this.updateAnnotationsList();
        this.setUnsavedChanges();
    }

    // 更新标注列表显示
    updateAnnotationsList() {
        const annotationDetails = document.getElementById('annotation-details');
        const annotationList = annotationDetails.querySelector('.annotation-list');
        // 保存滚动位置
        const scrollTop = annotationList ? annotationList.scrollTop : 0;

        annotationDetails.innerHTML = '';

        const currentImageName = this.images[this.currentImageIndex].name;
        const imageAnnotations = this.annotations[currentImageName] || [];

        if (imageAnnotations.length === 0) {
            const noAnnotationMsg = document.createElement('p');
            noAnnotationMsg.textContent = '当前图像没有标注';
            annotationDetails.appendChild(noAnnotationMsg);
            return;
        }

        const listTitle = document.createElement('h4');
        listTitle.textContent = '标注列表';
        annotationDetails.appendChild(listTitle);

        const list = document.createElement('div');
        list.className = 'annotation-list';
        annotationDetails.appendChild(list);

        imageAnnotations.forEach((polygon, index) => {
            const listItem = document.createElement('div');
            listItem.className = 'annotation-list-item';
            listItem.style.backgroundColor = this.getCategoryColor(polygon.classId);

            const annotationText = document.createElement('span');
            annotationText.textContent = `${index + 1}. ${polygon.classId}`;
            listItem.appendChild(annotationText);

            // 添加跳转到标注位置的点击事件
            listItem.addEventListener('click', (e) => {
                // 防止点击子按钮时触发
                if (e.target !== listItem && e.target !== annotationText) {
                    return;
                }
                
                // 计算标注的边界框
                let minX = Infinity, minY = Infinity;
                let maxX = -Infinity, maxY = -Infinity;
                
                polygon.points.forEach(point => {
                    minX = Math.min(minX, point[0]);
                    minY = Math.min(minY, point[1]);
                    maxX = Math.max(maxX, point[0]);
                    maxY = Math.max(maxY, point[1]);
                });
                
                // 计算边界框中心和大小
                const centerX = (minX + maxX) / 2;
                const centerY = (minY + maxY) / 2;
                const width = maxX - minX;
                const height = maxY - minY;
                
                // 计算合适的缩放比例，确保整个标注在画布中可见，并留有一些边距
                const canvasWidth = this.canvas.width;
                const canvasHeight = this.canvas.height;
                const margin = 50; // 边距
                
                const scaleX = (canvasWidth - margin * 2) / width;
                const scaleY = (canvasHeight - margin * 2) / height;
                const newZoom = Math.min(scaleX, scaleY, 5); // 限制最大缩放比例为5
                
                // 计算新的平移量，使标注中心位于画布中心
                this.zoom = newZoom;
                this.translateX = (canvasWidth / 2) - (centerX * this.zoom);
                this.translateY = (canvasHeight / 2) - (centerY * this.zoom);
                
                // 选中该多边形
                this.selectedPolygon = polygon;
                this.selectedVertex = null;
                this.redraw();
            });

            const editBtn = document.createElement('button');
            editBtn.textContent = '编辑';
            editBtn.className = 'btn btn-sm btn-secondary';
            editBtn.addEventListener('click', () => {
                this.selectedPolygon = polygon;
                this.selectedVertex = null;
                this.switchTool('edit-tool');
                this.redraw();
            });
            listItem.appendChild(editBtn);

            const deleteBtn = document.createElement('button');
            deleteBtn.textContent = '删除';
            deleteBtn.className = 'btn btn-sm btn-danger';
            deleteBtn.addEventListener('click', () => {
                // 在删除之前保存当前状态到历史记录
                this.saveStateToHistory();
                
                const polygonIndex = imageAnnotations.indexOf(polygon);
                imageAnnotations.splice(polygonIndex, 1);
                this.selectedPolygon = null;
                this.selectedVertex = null;
                this.redraw();
                this.updateAnnotationsList();
                this.setUnsavedChanges();
            });
            listItem.appendChild(deleteBtn);

            list.appendChild(listItem);
        });
        // 恢复滚动位置
        list.scrollTop = scrollTop;
    }

    // 删除选中的标注
    deleteSelectedAnnotation() {
        if (!this.selectedPolygon) return;
        
        const currentImageName = this.images[this.currentImageIndex].name;
        const imageAnnotations = this.annotations[currentImageName] || [];
        
        // 在删除之前保存当前状态
        this.saveStateToHistory();
        
        // 找到选中的标注
        const polygonIndex = imageAnnotations.indexOf(this.selectedPolygon);
        if (polygonIndex !== -1) {
            imageAnnotations.splice(polygonIndex, 1);
            this.selectedPolygon = null;
            this.selectedVertex = null;
        }
        
        this.redraw();
        this.updateAnnotationsList();
        this.setUnsavedChanges();
    }
    
    // 检查是否靠近第一个点（用于闭合多边形）
    isCloseToFirstPoint(x, y, tolerance = 10) {
        if (this.currentPolygon.length === 0) return false;
        
        const firstPoint = this.currentPolygon[0];
        const distance = Math.sqrt(Math.pow(x - firstPoint[0], 2) + Math.pow(y - firstPoint[1], 2));
        
        return distance < tolerance;
    }
    
    // 保存多边形
    savePolygon() {
        if (this.currentPolygon.length < 3) return;
        
        const currentImageName = this.images[this.currentImageIndex].name;
        
        // 在添加新多边形之前保存当前状态
        this.saveStateToHistory();
        
        // 确保标注对象存在
        if (!this.annotations[currentImageName]) {
            this.annotations[currentImageName] = [];
        }
        
        // 创建新的标注（与后端API格式一致）
        const newAnnotation = {
            id: Date.now(),
            classId: this.currentCategory,
            points: [...this.currentPolygon],
            timestamp: new Date().toISOString()
        };
        
        // 添加标注
        this.annotations[currentImageName].push(newAnnotation);
        
        // 更新文件列表
        this.updateFileList();
        this.updateStats();
        // 更新标注列表
        this.updateAnnotationsList();
        // 标记为未保存状态
        this.setUnsavedChanges();
    }
    
    // 找到距离给定点最近的多边形边缘
    findNearestEdge(x, y, tolerance = 10) {
        const currentImageName = this.images[this.currentImageIndex].name;
        const imageAnnotations = this.annotations[currentImageName] || [];
        
        let closestEdge = null;
        let closestPolygon = null;
        let minDistance = tolerance;
        let edgeIndex = -1;

        for (let polygon of imageAnnotations) {
            const points = polygon.points;
            const numPoints = points.length;
            
            for (let i = 0; i < numPoints; i++) {
                const j = (i + 1) % numPoints;
                const p1 = points[i];
                const p2 = points[j];
                
                // 计算点到线段的距离
                const distance = this.pointToLineDistance(x, y, p1[0], p1[1], p2[0], p2[1]);
                
                if (distance < minDistance) {
                    // 检查点是否在线段附近
                    const lineLength = Math.sqrt(Math.pow(p2[0] - p1[0], 2) + Math.pow(p2[1] - p1[1], 2));
                    const distToP1 = Math.sqrt(Math.pow(x - p1[0], 2) + Math.pow(y - p1[1], 2));
                    const distToP2 = Math.sqrt(Math.pow(x - p2[0], 2) + Math.pow(y - p2[1], 2));
                    
                    if (distToP1 <= lineLength + tolerance && distToP2 <= lineLength + tolerance) {
                        minDistance = distance;
                        closestEdge = [p1, p2];
                        closestPolygon = polygon;
                        edgeIndex = j; // 新顶点将插入到p1和p2之间，索引为j
                    }
                }
            }
        }
        
        return { edge: closestEdge, polygon: closestPolygon, edgeIndex: edgeIndex };
    }
    
    // 计算点到线段的距离
    pointToLineDistance(px, py, x1, y1, x2, y2) {
        const A = px - x1;
        const B = py - y1;
        const C = x2 - x1;
        const D = y2 - y1;
        
        const dot = A * C + B * D;
        const lenSq = C * C + D * D;
        let param = -1;
        
        if (lenSq !== 0) {
            param = dot / lenSq;
        }
        
        let xx, yy;
        
        if (param < 0) {
            xx = x1;
            yy = y1;
        } else if (param > 1) {
            xx = x2;
            yy = y2;
        } else {
            xx = x1 + param * C;
            yy = y1 + param * D;
        }
        
        const dx = px - xx;
        const dy = py - yy;
        
        return Math.sqrt(dx * dx + dy * dy);
    }
    
    // 选择顶点或多边形
    selectVertex(x, y, tolerance = 10) {
        const currentImageName = this.images[this.currentImageIndex].name;
        const imageAnnotations = this.annotations[currentImageName] || [];

        let closestVertex = null;
        let closestPolygon = null;
        let minDistance = tolerance;

        // 先尝试选择顶点，找到距离鼠标最近的顶点
        for (let polygon of imageAnnotations) {
            for (let vertex of polygon.points) {
                const distance = Math.sqrt(Math.pow(x - vertex[0], 2) + Math.pow(y - vertex[1], 2));
                if (distance < minDistance) {
                    closestVertex = vertex;
                    closestPolygon = polygon;
                    minDistance = distance;
                }
            }
        }

        // 如果找到最近的顶点，则选择它
        if (closestVertex) {
            this.selectedPolygon = closestPolygon;
            this.selectedVertex = closestVertex;
            return;
        }

        // 尝试选择多边形
        for (let polygon of imageAnnotations) {
            if (this.isPointInPolygon({x, y}, polygon.points)) {
                this.selectedPolygon = polygon;
                this.selectedVertex = null;
                return;
            }
        }

        // 未找到任何标注
        this.selectedPolygon = null;
        this.selectedVertex = null;
    }
    
    // 处理画布右键菜单事件 - 用于删除顶点
    handleCanvasContextMenu(e) {
        e.preventDefault(); // 阻止默认右键菜单
        
        if (this.currentTool !== 'edit-tool') return;
        
        const rect = this.canvas.getBoundingClientRect();
        const scaleX = this.canvas.width / rect.width;
        const scaleY = this.canvas.height / rect.height;
        const canvasX = (e.clientX - rect.left) * scaleX;
        const canvasY = (e.clientY - rect.top) * scaleY;
        const x = (canvasX - this.translateX) / this.zoom;
        const y = (canvasY - this.translateY) / this.zoom;
        
        // 尝试选择并删除顶点
        this.selectVertex(x, y);
        
        if (this.selectedVertex && this.selectedPolygon) {
                const vertexIndex = this.selectedPolygon.points.indexOf(this.selectedVertex);
                
                // 确保删除后多边形至少有3个点（保持多边形有效性）
                if (this.selectedPolygon.points.length > 3) {
                    // 在删除顶点之前保存当前状态
                    this.saveStateToHistory();
                    
                    // 删除顶点
                    this.selectedPolygon.points.splice(vertexIndex, 1);
                    this.selectedVertex = null;
                    this.redraw();
                    this.setUnsavedChanges();
                    console.log('顶点已删除');
                } else {
                    alert('多边形至少需要3个顶点，无法删除！');
                }
        }
    };




    // 绘制多边形
    drawPolygon(polygon, isSelected = false) {
        // 根据类别设置颜色
        const color = this.getCategoryColor(polygon.classId);
        const isCurrentSelected = this.selectedPolygon === polygon;
        const borderWidth = isCurrentSelected ? 3 : 2;

        // 绘制多边形填充
        this.ctx.beginPath();
        this.ctx.moveTo(polygon.points[0][0], polygon.points[0][1]);
        for (let i = 1; i < polygon.points.length; i++) {
            this.ctx.lineTo(polygon.points[i][0], polygon.points[i][1]);
        }
        this.ctx.closePath();
        this.ctx.fillStyle = color;
        this.ctx.globalAlpha = 0.3;
        this.ctx.fill();
        this.ctx.globalAlpha = 1;

        // 绘制多边形边框
        this.ctx.strokeStyle = isCurrentSelected ? '#ffffff' : color;
        this.ctx.lineWidth = (isCurrentSelected ? borderWidth + 1 : borderWidth) / this.zoom;
        this.ctx.stroke();
        this.ctx.strokeStyle = color;
        this.ctx.lineWidth = borderWidth / this.zoom;
        this.ctx.stroke();

        // 绘制顶点
        polygon.points.forEach(point => {
            const isVertexSelected = this.selectedVertex === point;
            const vertexColor = isVertexSelected ? '#ffffff' : color;
            const vertexSize = isVertexSelected ? 3 : 2;
            this.drawPoint(point[0], point[1], vertexColor, vertexSize, true); // 传入isTransformed=true，因为上下文已经应用了缩放和平移
        });
    };

    // 绘制点
    drawPoint(x, y, color, size = 3, isTransformed = false) {
        let screenX, screenY, radius, lineWidth;
        
        if (isTransformed) {
            // 上下文已经应用了缩放和平移，直接使用坐标
            screenX = x;
            screenY = y;
            // 调整半径和线宽以适应缩放
            radius = size / this.zoom;
            lineWidth = 2 / this.zoom;
        } else {
            // 应用缩放和平移变换
            screenX = (x * this.zoom) + this.translateX;
            screenY = (y * this.zoom) + this.translateY;
            radius = size;
            lineWidth = 2;
        }
        
        this.ctx.beginPath();
        this.ctx.arc(screenX, screenY, radius, 0, 2 * Math.PI);
        this.ctx.fillStyle = color;
        this.ctx.fill();
        this.ctx.strokeStyle = '#ffffff';
        this.ctx.lineWidth = lineWidth;
        this.ctx.stroke();
    };
    
    // 添加自定义类别
    addCategory() {
        const categoryName = prompt('请输入新类别的名称:');
        if (!categoryName) return;
        
        const categoryValue = categoryName.toLowerCase().replace(/\s+/g, '_');
        const color = prompt('请输入颜色 (格式: #RRGGBB):');
        if (!color) return;
        
        // 检查颜色格式
        if (!/^#[0-9A-Fa-f]{6}$/.test(color)) {
            alert('颜色格式无效，请使用#RRGGBB格式');
            return;
        }
        
        // 检查是否已存在相同类别的选项
        const existingOption = [...this.categorySelect.children].find(opt => opt.dataset.value === categoryValue);
        if (!existingOption) {
            // 创建类别列表项并添加到容器
            const categoryItem = document.createElement('div');
            categoryItem.className = 'category-item';
            categoryItem.dataset.value = categoryValue;
            categoryItem.textContent = categoryName;
            this.categorySelect.appendChild(categoryItem);
        }
        
        // 保存颜色
        this.categoryColors[categoryValue] = color;
        
        alert('类别已添加!');
    };
    
    // 导入类别
    handleImportCategories(e) {
        const file = e.target.files[0];
        if (!file) return;
        
        const reader = new FileReader();
        reader.onload = (event) => {
            try {
                const categories = JSON.parse(event.target.result);
                
                // 添加所有导入的类别
                categories.forEach(category => {
                    if (category.name && category.value && category.color) {
                        // 检查是否已存在
                        const existingOption = [...this.categorySelect.children].find(opt => opt.dataset.value === category.value);
                        if (!existingOption) {
                            // 创建类别列表项并添加到容器
                            const categoryItem = document.createElement('div');
                            categoryItem.className = 'category-item';
                            categoryItem.dataset.value = category.value;
                            categoryItem.textContent = category.name || category.value;
                            this.categorySelect.appendChild(categoryItem);
                        }
                        
                        // 更新颜色
                        this.categoryColors[category.value] = category.color;
                    }
                });
                
                alert('类别导入成功!');
                this.importCategoriesFile.value = ''; // 重置文件输入
            } catch (error) {
                alert('导入失败: ' + error.message);
            }
        };
        
        reader.readAsText(file);
    };
    
    // 加载配置文件
    loadConfigFile() {
        this.configFileInput.click();
    }
    
    // 处理配置文件选择
    handleConfigFile(e) {
        const file = e.target.files[0];
        if (!file) return;
        
        // 本地处理配置文件
        const reader = new FileReader();
        reader.onload = (event) => {
            try {
                // 简单解析YAML格式（基本的键值对解析）
                const content = event.target.result;
                const lines = content.split('\n');
                const categories = [];
                
                // 简单解析labels部分（适配apollo_config.yaml格式）
                let inLabelsSection = false;
                for (const line of lines) {
                    const trimmedLine = line.trim();
                    if (trimmedLine === 'labels:') {
                        inLabelsSection = true;
                        continue;
                    }
                    if (inLabelsSection && trimmedLine.startsWith('-')) {
                        const name = trimmedLine.substring(1).trim();
                        if (name && !name.startsWith('#')) { // 忽略注释
                            categories.push(name);
                        }
                    }
                    // 遇到其他section结束labels解析
                    if (inLabelsSection && trimmedLine && !trimmedLine.startsWith('-') && !trimmedLine.startsWith('#') && trimmedLine.includes(':')) {
                        inLabelsSection = false;
                    }
                }
                
                // 如果没有解析到labels，尝试解析categories部分
                if (categories.length === 0) {
                    let inCategoriesSection = false;
                    for (const line of lines) {
                        const trimmedLine = line.trim();
                        if (trimmedLine === 'categories:') {
                            inCategoriesSection = true;
                            continue;
                        }
                        if (inCategoriesSection && trimmedLine.startsWith('-')) {
                            const name = trimmedLine.substring(1).trim();
                            if (name && !name.startsWith('#')) {
                                categories.push(name);
                            }
                        }
                        // 遇到其他section结束categories解析
                        if (inCategoriesSection && trimmedLine && !trimmedLine.startsWith('-') && !trimmedLine.startsWith('#') && trimmedLine.includes(':')) {
                            inCategoriesSection = false;
                        }
                    }
                }
                
                // 如果没有解析到categories，尝试直接解析JSON格式
                if (categories.length === 0) {
                    try {
                        const jsonData = JSON.parse(content);
                        if (jsonData.categories && Array.isArray(jsonData.categories)) {
                            jsonData.categories.forEach(cat => {
                                if (typeof cat === 'string') {
                                    categories.push(cat);
                                } else if (cat.name) {
                                    categories.push(cat.name);
                                }
                            });
                        }
                    } catch (jsonError) {
                        console.warn('配置文件不是有效的JSON格式');
                    }
                }
                
                // 处理类别数据
                if (categories.length === 0) {
                    alert('未在配置文件中找到有效类别数据');
                    return;
                }
                
                // 清空现有类别
                this.categorySelect.innerHTML = '';
                this.categoryColors = {};
                this.categories = [];
                
                // 添加新类别（跳过ID为0的背景类别）
                let validCategoriesCount = 0;
                categories.forEach((category, index) => {
                    // 为每个类别生成默认颜色
                    const generateColor = (idx) => {
                        const colors = ['#FF5733', '#33FF57', '#3357FF', '#F3FF33', '#FF33F3', '#33FFF3', '#FF8C33', '#8C33FF', '#33FF8C', '#FF3333', '#33FF33', '#3333FF', '#FFFF33', '#FF33FF', '#33FFFF'];
                        return colors[idx % colors.length];
                    };
                    
                    // 计算类别ID（从1开始，跳过0）
                    const categoryId = index + 1;
                    
                    // 创建类别列表项并添加到容器
                    const categoryItem = document.createElement('div');
                    categoryItem.className = 'category-item';
                    categoryItem.dataset.value = category;
                    categoryItem.dataset.id = categoryId;
                    categoryItem.textContent = category;
                    this.categorySelect.appendChild(categoryItem);
                    
                    // 设置类别颜色
                    this.categoryColors[category] = generateColor(validCategoriesCount);
                    
                    // 添加到categories数组
                    this.categories.push({
                        id: categoryId,
                        name: category
                    });
                    
                    validCategoriesCount++;
                });
                
                // 设置默认选中第一个类别
                if (this.categorySelect.children.length > 0) {
                    // 为第一个类别添加active类
                    this.categorySelect.children[0].classList.add('active');
                    // 设置当前类别
                    this.currentCategory = this.categorySelect.children[0].dataset.value;
                }
                
                alert('配置加载成功！有效类别数量: ' + validCategoriesCount);
                this.configFileInput.value = ''; // 重置文件输入
            } catch (error) {
                alert('加载配置失败: ' + error.message);
                console.error('处理配置文件时出错:', error);
            }
        };
        reader.readAsText(file);
    }
    
    // 获取类别颜色
    getCategoryColor(category) {
        return this.categoryColors[category] || '#808080';
    };
    
    // 绘制所有标注
    drawAnnotations(imageName) {
        const imageAnnotations = this.annotations[imageName] || [];
        
        imageAnnotations.forEach(polygon => {
            this.drawPolygon(polygon);
        });
    };
    
    // 重绘画布
    redraw() {
        if (this.currentImageIndex < 0 || this.currentImageIndex >= this.images.length) return;
        
        const currentImage = this.images[this.currentImageIndex];
        const img = currentImage.imageObject;
        
        // 清除画布
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        // 保存当前上下文状态
        this.ctx.save();
        
        // 应用缩放和平移变换
        this.ctx.translate(this.translateX, this.translateY);
        this.ctx.scale(this.zoom, this.zoom);
        
        // 绘制图像
        this.ctx.drawImage(img, 0, 0, this.canvas.width, this.canvas.height);
        
        // 绘制标注
        this.drawAnnotations(currentImage.name);
        
        // 绘制当前正在绘制的临时多边形
        if (this.isDrawing && this.currentPolygon.length > 0) {
                // 绘制当前多边形
                if (this.currentPolygon.length > 1) {
                    this.ctx.beginPath();
                    this.ctx.moveTo(this.currentPolygon[0][0], this.currentPolygon[0][1]);
                    for (let i = 1; i < this.currentPolygon.length; i++) {
                        this.ctx.lineTo(this.currentPolygon[i][0], this.currentPolygon[i][1]);
                    }
                this.ctx.strokeStyle = '#ff0000';
                this.ctx.lineWidth = 2 / this.zoom;
                this.ctx.stroke();
            }
            
            // 绘制所有点 (应用缩放和平移变换)
            this.currentPolygon.forEach(point => {
                this.drawPoint(point[0], point[1], '#ff0000', 3, true);
            });
        }
        
        // 恢复上下文状态
        this.ctx.restore();
    };
    
    // 选择工具
    selectTool(e) {
        // 移除所有工具的激活状态
        this.toolButtons.forEach(btn => btn.classList.remove('active'));
        
        // 设置当前工具
        this.currentTool = e.target.id;
        e.target.classList.add('active');
        
        // 重置状态
        this.isDrawing = false;
        this.currentPolygon = [];
        this.selectedPolygon = null;
        this.selectedVertex = null;
        this.isAddingVertex = false;

        // 更改鼠标指针形态
        this.updateCursorStyle();
    };
    
    // 切换工具
    switchTool(toolId) {
        // 找到对应的工具按钮
        const toolBtn = document.getElementById(toolId);
        if (toolBtn) {
            // 移除所有工具的激活状态
            this.toolButtons.forEach(btn => btn.classList.remove('active'));
            
            // 设置当前工具
            this.currentTool = toolId;
            toolBtn.classList.add('active');
            
            // 重置状态
            this.isDrawing = false;
            this.currentPolygon = [];
            this.selectedPolygon = null;
            this.selectedVertex = null;
            this.isAddingVertex = false;

            // 更改鼠标指针形态
            this.updateCursorStyle();
        }
        this.redraw();
    };
    
    // 保存标注 (LabelMe格式)
    saveAnnotations() {
        if (this.images.length === 0) {
            alert('没有图像需要保存标注！');
            return;
        }

        const currentImage = this.images[this.currentImageIndex];
        const currentImageName = currentImage.name;
        const annotationData = this.annotations[currentImageName] || [];

        // 保存为LabelMe格式的JSON（保留兼容性）
        const labelmeJSON = {
            "version": "5.0.1",
            "flags": {},
            "shapes": annotationData.map(annotation => ({
                "label": annotation.classId,
                "points": annotation.points,
                "group_id": null,
                "shape_type": "polygon",
                "flags": {},
                "description": ""
            })),
            "imagePath": currentImage.filePath,
            "imageData": null, // 移除base64图像数据以避免413错误
            "imageHeight": this.canvas.height,
            "imageWidth": this.canvas.width
        };

        // 准备FormData发送JSON格式标注
        const formData = new FormData();
        formData.append('image_filepath', currentImage.filePath);
        formData.append('image_filename', currentImageName);
        
        // 提取并添加图像目录信息，正确处理不同的路径分隔符
        let imageDirectory = '';
        if (currentImage.filePath) {
            const lastSlashIndex = Math.max(
                currentImage.filePath.lastIndexOf('/'),
                currentImage.filePath.lastIndexOf('\\')
            );
            if (lastSlashIndex !== -1) {
                imageDirectory = currentImage.filePath.substring(0, lastSlashIndex + 1);
            }
        }
        console.log('图像目录:', imageDirectory);
        formData.append('image_directory', imageDirectory);
        
        formData.append('annotation_data', JSON.stringify(labelmeJSON));
        
        // 发送到后端保存
        fetch('http://localhost:5000/api/save_annotation', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            console.log('后端保存响应:', data);
            if (data.error) {
                console.error('保存标注失败:', data.error);
                alert('保存标注失败: ' + data.error);
            } else {
                console.log('标注保存成功:', data.message, 'JSON路径:', data.annotation_json_path || '未提供');
                // 显示保存成功的消息和路径信息
                if (data.annotation_json_path) {
                    console.log('标注文件保存在:', data.annotation_json_path);
                }
                this.hasUnsavedChanges = false;
                this.lastSaveTime = Date.now();
                // 重要：保存成功后更新文件列表中的标注状态
                this.updateFileList();
            }
        })
        .catch(error => {
            console.error('保存标注时发生错误:', error);
            alert('保存标注失败: ' + error.message);
        });
    };

    // 获取当前图像的Base64编码
    getCurrentImageBase64() {
        const currentImage = this.images[this.currentImageIndex];
        const canvas = document.createElement('canvas');
        canvas.width = currentImage.imageObject.width;
        canvas.height = currentImage.imageObject.height;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(currentImage.imageObject, 0, 0);
        return canvas.toDataURL('image/png');
    };
    
    // 自动标注
    autoAnnotate() {
        // 显示加载中提示
        console.log('开始自动标注，调用API...');
        alert('正在进行自动标注，请稍候...');
        
        // 获取当前显示的图片
        if (this.currentImageIndex < 0 || this.currentImageIndex >= this.images.length || !this.images[this.currentImageIndex]?.imageObject) {
            alert('请先加载图片再进行自动标注');
            return;
        }
        
        // 创建FormData对象来上传图片
        const formData = new FormData();
        
        // 将当前canvas转换为Blob对象
        this.canvas.toBlob((blob) => {
            // 添加图片到FormData
            formData.append('image', blob, 'current_image.png');
            
            // 添加模型选择和置信度参数
            const selectedModel = this.modelSelect?.value || 'models/sam_finetuned.pth';
            const confidenceThreshold = parseFloat(this.confidenceValue?.textContent || '0.8');
            
            formData.append('model_path', selectedModel);
            formData.append('model_type', 'vit_l');
            formData.append('confidence_threshold', confidenceThreshold.toString());
            
            // 发送POST请求到后端API
            fetch('http://localhost:5000/api/auto_annotate_single', {
                method: 'POST',
                body: formData
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                if (data.success) {
                    // 清空现有标注
                    this.clearAnnotations();
                    
                    // 添加自动生成的多边形标注
                    data.polygons.forEach(polygon => {
                        // 查找对应的类别ID
                        let classId = polygon.classId;
                        // 如果classId是字符串，尝试转换为数字ID
                        if (typeof classId === 'string') {
                            // 查找或创建对应的类别
                            const category = this.categories.find(cat => cat.name === classId);
                            if (category) {
                                classId = category.id;
                            } else {
                                // 默认使用第一个非背景类别
                                const nonBackgroundCategory = this.categories.find(cat => cat.id !== 0);
                                if (nonBackgroundCategory) {
                                    classId = nonBackgroundCategory.id;
                                } else {
                                    classId = 1; // 默认类别ID
                                }
                            }
                        }
                        
                        // 创建新的多边形标注 - 注意：points格式需要是[[x1,y1], [x2,y2], ...]，而不是[x1,y1,x2,y2,...]
                        const points = [];
                        for (let i = 0; i < polygon.points.length; i += 2) {
                            points.push([polygon.points[i], polygon.points[i+1]]);
                        }
                        
                        const annotation = {
                            id: this.generateUniqueId(),
                            points: points,
                            classId: classId,
                            fillColor: polygon.fillColor || 'rgba(255, 0, 0, 0.3)',
                            strokeColor: '#ff0000',
                            strokeWidth: 2,
                            closed: true,
                            selected: false
                        };
                        
                        // 添加到当前图像的标注中
                        const currentImageName = this.currentImage.name;
                        if (!this.annotations[currentImageName]) {
                            this.annotations[currentImageName] = [];
                        }
                        this.annotations[currentImageName].push(annotation);
                    });
                    
                    // 重新渲染标注
                    this.redraw();
                    // 更新标注列表
                    this.updateFileList();
                    this.updateAnnotationsList();
                    
                    alert(`成功生成 ${data.polygons.length} 个标注`);
                } else {
                    throw new Error(data.error || '自动标注失败');
                }
            })
            .catch(error => {
                console.error('自动标注错误:', error);
                alert(`自动标注失败: ${error.message}`);
            });
        }, 'image/png');
    };
    
    // 显示上一张图像
    showPreviousImage() {
        if (this.currentImageIndex > 0) {
            this.currentImageIndex--;
            this.loadImage(this.images[this.currentImageIndex]);
            this.updateFileList();
        }
    };
    
    // 显示下一张图像
    showNextImage() {
        if (this.currentImageIndex < this.images.length - 1) {
            this.currentImageIndex++;
            this.loadImage(this.images[this.currentImageIndex]);
            this.updateFileList();
        }
    };
    
    // 更新统计信息
    updateStats() {
        const total = this.images.length;
        let annotated = 0;
        
        this.images.forEach(image => {
            if (this.annotations[image.name] && this.annotations[image.name].length > 0) {
                annotated++;
            }
        });
        
        document.getElementById('annotated-count').textContent = annotated;
        document.getElementById('unannotated-count').textContent = total - annotated;
        document.getElementById('total-count').textContent = total;
    };
    
    // 更新状态栏
    updateStatusBar(imageName) {
        document.getElementById('current-file-info').textContent = `当前图像: ${imageName}`;
        document.getElementById('canvas-info').textContent = `画布: ${this.canvas.width}x${this.canvas.height}`;
    };
    
    // 调整画布大小
    resizeCanvas() {
        // 这里可以实现画布的自适应调整
    };
    
    // 初始化自动保存定时器
    initAutoSave() {
        this.autoSaveTimer = setInterval(() => {
            if (this.hasUnsavedChanges) {
                const now = Date.now();
                if (now - this.lastSaveTime > this.autoSaveInterval) {
                    this.saveAnnotations();
                    this.hasUnsavedChanges = false;
                    this.lastSaveTime = now;
                    console.log('自动保存完成');
                    // 可以添加一个提示给用户，比如状态栏显示
                    const statusBar = document.createElement('div');
                    statusBar.id = 'auto-save-status';
                    statusBar.textContent = '已自动保存';
                    statusBar.style.position = 'fixed';
                    statusBar.style.bottom = '10px';
                    statusBar.style.right = '10px';
                    statusBar.style.backgroundColor = 'rgba(0, 0, 0, 0.8)';
                    statusBar.style.color = 'white';
                    statusBar.style.padding = '10px';
                    statusBar.style.borderRadius = '5px';
                    document.body.appendChild(statusBar);
                    setTimeout(() => statusBar.remove(), 2000);
                }
            }
        }, 1000); // 每秒检查一次
    };

    // 更新鼠标指针样式
    updateCursorStyle() {
        const canvas = document.getElementById('annotation-canvas');
        switch (this.currentTool) {
            case 'draw-tool':
                canvas.style.cursor = 'crosshair';
                break;
            case 'pan-tool':
                canvas.style.cursor = 'move';
                break;
            case 'edit-tool':
                canvas.style.cursor = this.isAddingVertex ? 'crosshair' : 'pointer';
                break;
            case 'delete-tool':
                canvas.style.cursor = 'not-allowed';
                break;
            default:
                canvas.style.cursor = 'default';
        }
    };

    // 设置未保存状态
    setUnsavedChanges() {
        this.hasUnsavedChanges = true;
    };
    
    // 记录当前状态到历史记录
    saveStateToHistory() {
        // 只有当有当前图像时才保存状态
        if (this.currentImageIndex === -1 || !this.images[this.currentImageIndex]) {
            return;
        }
        
        const currentImageName = this.images[this.currentImageIndex].name;
        
        // 确保当前图像的历史记录数组和索引存在
        if (!this.undoHistory[currentImageName]) {
            this.undoHistory[currentImageName] = [];
        }
        if (this.undoHistoryIndex[currentImageName] === undefined) {
            this.undoHistoryIndex[currentImageName] = -1;
        }
        
        // 创建当前状态的深拷贝
        const currentState = {
            currentImageName: currentImageName,
            annotations: JSON.parse(JSON.stringify(this.annotations[currentImageName] || [])),
            timestamp: Date.now()
        };
        
        // 如果当前索引不是最新的，则清除后续的历史记录
        if (this.undoHistoryIndex[currentImageName] < this.undoHistory[currentImageName].length - 1) {
            this.undoHistory[currentImageName] = this.undoHistory[currentImageName].slice(0, this.undoHistoryIndex[currentImageName] + 1);
        }
        
        // 添加新的历史记录
        this.undoHistory[currentImageName].push(currentState);
        this.undoHistoryIndex[currentImageName] = this.undoHistory[currentImageName].length - 1;
        
        // 限制历史记录数量
        if (this.undoHistory[currentImageName].length > this.maxHistorySize) {
            this.undoHistory[currentImageName].shift(); // 移除最旧的记录
            this.undoHistoryIndex[currentImageName]--;
        }
    };
    
    // 撤销操作
    undo() {
        const currentImageName = this.images[this.currentImageIndex].name;
        
        // 确保当前图像的历史记录存在
        if (!this.undoHistory[currentImageName] || this.undoHistoryIndex[currentImageName] <= 0) {
            return; // 没有历史记录或已经是最早的状态，无法撤销
        }
        
        // 移动到上一个历史记录
        this.undoHistoryIndex[currentImageName]--;
        const previousState = this.undoHistory[currentImageName][this.undoHistoryIndex[currentImageName]];
        
        // 恢复到上一个状态
        if (previousState.currentImageName === currentImageName) {
            // 深拷贝恢复标注数据
            this.annotations[currentImageName] = JSON.parse(JSON.stringify(previousState.annotations));
            
            // 清除选择状态
            this.selectedPolygon = null;
            this.selectedVertex = null;
            
            // 重新绘制画布
            this.redraw();
            // 更新标注列表
            this.updateAnnotationsList();
            // 设置为未保存状态
            this.setUnsavedChanges();
        }
    };
}

// 页面加载完成后初始化
window.addEventListener('DOMContentLoaded', () => {
    new ImageAnnotationTool();
});