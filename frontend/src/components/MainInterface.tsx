import React, { useState, useRef, useEffect, useCallback } from 'react';
import { Layout, Button, Upload, message, Select, Card, Space, Tabs } from 'antd';
import { UploadOutlined, SaveOutlined, RobotOutlined, ImportOutlined, FolderOpenOutlined, DeleteOutlined, EditOutlined, FolderOutlined, ZoomInOutlined, ZoomOutOutlined, UndoOutlined } from '@ant-design/icons';
import AnnotationCanvas from './AnnotationCanvas';
import DirectoryBrowser from './DirectoryBrowser';
import axios from 'axios';

const { Sider, Content } = Layout;
const { Option } = Select;
const { TabPane } = Tabs;

interface ImageInfo {
  filename: string;
  filepath: string;
}

interface ClassInfo {
  [key: number]: string;
}

interface ColorInfo {
  [key: number]: number[];
}

interface Polygon {
  id: string;
  points: { x: number; y: number }[];
  classId: number;
}

const MainInterface: React.FC = () => {
  const [images, setImages] = useState<ImageInfo[]>([]);
  const [selectedImage, setSelectedImage] = useState<ImageInfo | null>(null);
  const [classes, setClasses] = useState<ClassInfo>({});
  const [colors, setColors] = useState<ColorInfo>({});
  const [selectedClass, setSelectedClass] = useState<number>(0);
  const [selectedModel, setSelectedModel] = useState<string>('deeplab');
  const [loading, setLoading] = useState(false);
  const [aiLoading, setAiLoading] = useState(false);
  const canvasRef = useRef<any>(null);

  const loadClasses = useCallback(async () => {
    try {
      const response = await axios.get('/api/classes');
      setClasses(response.data.classes);
      setColors(response.data.colors);
    } catch (error) {
      // 尝试加载默认配置
      loadDefaultConfig();
    }
  }, []);

  const loadImages = async () => {
    try {
      const response = await axios.get('/api/images');
      setImages(response.data.images);
    } catch (error) {
      message.error('加载图像列表失败');
    }
  };

  useEffect(() => {
    loadImages();
    loadClasses();
  }, [loadClasses]);

  const loadDefaultConfig = () => {
    // 默认Apollo配置
    const defaultClasses: ClassInfo = {
      0: 'background',
      1: 's_w_d',
      2: 's_y_d',
      3: 'ds_y_dn',
      4: 'sb_w_do',
      5: 'sb_y_do',
      6: 'b_w_g',
      7: 's_w_s',
      8: 's_w_c',
      9: 's_w_p',
      10: 'c_wy_z',
      11: 'a_w_c',
      12: 'b_n_sr',
      13: 'd_wy_za',
      14: 'r_wy_np',
      15: 'vom_c_n'
    };

    const defaultColors: ColorInfo = {
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
    };

    setClasses(defaultClasses);
    setColors(defaultColors);
  };

  const handleImageUpload = async (info: any) => {
    const { file } = info;
    const formData = new FormData();
    formData.append('image', file);

    try {
      setLoading(true);
      await axios.post('/api/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      message.success('图像上传成功');
      loadImages();
    } catch (error) {
      message.error('图像上传失败');
    } finally {
      setLoading(false);
    }
  };

  const handleDirectoryUpload = async (info: any) => {
    const { files } = info;
    
    for (const file of files) {
      if (file.type.startsWith('image/')) {
        const formData = new FormData();
        formData.append('image', file);
        
        try {
          await axios.post('/api/upload', formData, {
            headers: {
              'Content-Type': 'multipart/form-data',
            },
          });
        } catch (error) {
          console.error(`上传 ${file.name} 失败:`, error);
        }
      }
    }
    
    message.success('目录图像上传完成');
    loadImages();
  };

  const handleImageSelect = (image: ImageInfo) => {
    setSelectedImage(image);
    loadAnnotation(image.filename);
  };

  const loadAnnotation = async (filename: string) => {
    try {
      const response = await axios.get('/api/load_annotation', {
        params: { image_filename: filename }
      });
      
      if (response.data.annotation && canvasRef.current) {
        canvasRef.current.loadAnnotation(response.data.annotation);
      }
    } catch (error) {
      console.error('加载标注失败:', error);
    }
  };

  const handleSaveAnnotation = async () => {
    if (!selectedImage || !canvasRef.current) {
      message.error('请先选择图像');
      return;
    }

    try {
      const annotationData = canvasRef.current.getAnnotationData();
      await axios.post('/api/save_annotation', {
        image_filename: selectedImage.filename,
        annotation_data: annotationData
      });
      message.success('标注保存成功');
    } catch (error) {
      message.error('标注保存失败');
    }
  };

  const handleAIPredict = async () => {
    if (!selectedImage) {
      message.error('请先选择图像');
      return;
    }

    try {
      setAiLoading(true);
      const response = await axios.post('/api/predict', {
        image_filename: selectedImage.filename,
        model: selectedModel
      });
      
      if (canvasRef.current) {
        canvasRef.current.loadPrediction(response.data.mask);
      }
      message.success('AI预测完成');
    } catch (error) {
      message.error('AI预测失败');
      console.error('AI预测失败:', error);
    } finally {
      setAiLoading(false);
    }
  };

  const handleTrainModel = async () => {
    try {
      setLoading(true);
      const annotatedImages = images.filter(img => {
        // 检查是否存在对应的标注文件
        return true; // 简化处理
      });

      await axios.post('/api/train', {
        annotated_images: annotatedImages,
        classes: classes,
        colors: colors
      });
      message.success('模型训练完成');
    } catch (error) {
      message.error('模型训练失败');
    } finally {
      setLoading(false);
    }
  };

  const handleImportConfig = async (info: any) => {
    const { file } = info;
    const reader = new FileReader();
    
    reader.onload = (e) => {
      try {
        const content = e.target?.result as string;
        const config = parseYamlConfig(content);
        
        if (config.labels && config.label_colors) {
          const newClasses: ClassInfo = {};
          const newColors: ColorInfo = {};
          
          config.labels.forEach((label: string, index: number) => {
            newClasses[index] = label;
            if (config.label_colors[label]) {
              newColors[index] = config.label_colors[label];
            }
          });
          
          setClasses(newClasses);
          setColors(newColors);
          message.success('配置文件导入成功');
        }
      } catch (error) {
        message.error('配置文件解析失败');
      }
    };
    
    reader.readAsText(file);
  };

  const parseYamlConfig = (content: string) => {
    // 简单的YAML解析器
    const lines = content.split('\n');
    const config: any = { labels: [], label_colors: {} };
    
    let currentSection = '';
    lines.forEach(line => {
      const trimmed = line.trim();
      if (trimmed.startsWith('labels:')) {
        currentSection = 'labels';
      } else if (trimmed.startsWith('label_colors:')) {
        currentSection = 'colors';
      } else if (trimmed.startsWith('- ') && currentSection === 'labels') {
        config.labels.push(trimmed.substring(2));
      } else if (trimmed.includes(':') && currentSection === 'colors') {
        const [key, value] = trimmed.split(':').map(s => s.trim());
        if (value.startsWith('[') && value.endsWith(']')) {
          config.label_colors[key] = JSON.parse(value);
        }
      }
    });
    
    return config;
  };

  const handlePolygonClick = (polygon: Polygon) => {
    setSelectedClass(polygon.classId);
  };

  const handleDeleteSelected = () => {
    if (canvasRef.current) {
      canvasRef.current.deleteSelectedPolygon();
    }
  };

  const handleChangeClass = () => {
    if (canvasRef.current) {
      canvasRef.current.changePolygonClass(selectedClass);
    }
  };

  const handleClearAll = () => {
    if (canvasRef.current) {
      canvasRef.current.clearAnnotations();
    }
  };

  const handleCancelCurrentPolygon = () => {
    if (canvasRef.current) {
      canvasRef.current.clearCurrentPolygon();
    }
  };

  const handleFileSelect = async (filename: string, annotationPath?: string) => {
    // 选择从目录浏览器加载的文件
    const newImage: ImageInfo = {
      filename,
      filepath: `/uploads/${filename}`
    };
    
    setSelectedImage(newImage);
    
    // 刷新图像列表
    await loadImages();
    
    // 加载标注
    if (annotationPath) {
      try {
        const response = await axios.post('/api/load_annotation_from_directory', {
          image_path: annotationPath,
          annotation_path: annotationPath
        });
        
        if (response.data.annotation && canvasRef.current) {
          canvasRef.current.loadAnnotation(response.data.annotation);
        }
      } catch (error) {
        console.error('加载目录标注失败:', error);
        // 尝试从标准位置加载标注
        loadAnnotation(filename);
      }
    } else {
      loadAnnotation(filename);
    }
  };

  return (
    <Layout style={{ height: '100%' }}>
      <Sider width={320} style={{ background: '#fff', padding: '20px', overflowY: 'auto' }}>
        <Space direction="vertical" style={{ width: '100%' }} size="large">
          <Tabs defaultActiveKey="upload" size="small">
            <TabPane tab={<span><UploadOutlined />上传管理</span>} key="upload">
              <Space direction="vertical" style={{ width: '100%' }}>
                <Upload
                  accept="image/*"
                  beforeUpload={() => false}
                  onChange={handleImageUpload}
                  showUploadList={false}
                >
                  <Button icon={<UploadOutlined />} loading={loading}>
                    上传图像
                  </Button>
                </Upload>
                
                <Upload
                  directory
                  beforeUpload={() => false}
                  onChange={handleDirectoryUpload}
                  showUploadList={false}
                >
                  <Button icon={<FolderOpenOutlined />}>
                    导入目录
                  </Button>
                </Upload>
              </Space>
              
              <div style={{ marginTop: 10, maxHeight: 200, overflowY: 'auto' }}>
                {images.map((image) => (
                  <div
                    key={image.filename}
                    style={{
                      padding: '5px',
                      cursor: 'pointer',
                      background: selectedImage?.filename === image.filename ? '#e6f7ff' : 'transparent',
                      borderRadius: '4px'
                    }}
                    onClick={() => handleImageSelect(image)}
                  >
                    {image.filename}
                  </div>
                ))}
              </div>
            </TabPane>
            
            <TabPane tab={<span><FolderOutlined />目录浏览</span>} key="browse">
              <DirectoryBrowser onFileSelect={handleFileSelect} />
            </TabPane>
          </Tabs>
        </Space>
      </Sider>

      <Content style={{ padding: '20px', background: '#fff', position: 'relative' }}>
        {selectedImage ? (
          <div style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
            <div style={{ marginBottom: 10, fontSize: '16px', fontWeight: 'bold' }}>
              当前图像: {selectedImage.filename}
            </div>
            <div style={{ flex: 1, position: 'relative' }}>
              <AnnotationCanvas
                ref={canvasRef}
                imagePath={`/uploads/${selectedImage.filename}`}
                selectedClass={selectedClass}
                colors={colors}
                onPolygonClick={handlePolygonClick}
              />
            </div>
          </div>
        ) : (
          <div style={{ 
            display: 'flex', 
            justifyContent: 'center', 
            alignItems: 'center', 
            height: '100%',
            fontSize: '18px',
            color: '#999'
          }}>
            请选择或上传图像开始标注
          </div>
        )}
      </Content>

      <Sider width={320} style={{ background: '#fff', padding: '20px', overflowY: 'auto' }}>
        <Space direction="vertical" style={{ width: '100%' }} size="large">
          <Card title="标注工具" size="small">
            <div style={{ marginBottom: 10 }}>
              <label>选择类别: </label>
              <div style={{ 
                maxHeight: '200px', 
                overflowY: 'auto', 
                border: '1px solid #d9d9d9', 
                borderRadius: '6px',
                padding: '4px'
              }}>
                {Object.entries(classes).map(([id, name]) => (
                  <div
                    key={id}
                    style={{
                      padding: '8px 12px',
                      cursor: 'pointer',
                      backgroundColor: selectedClass === parseInt(id) ? '#1890ff' : 'transparent',
                      color: selectedClass === parseInt(id) ? '#fff' : '#000',
                      borderRadius: '4px',
                      marginBottom: '2px',
                      display: 'flex',
                      alignItems: 'center',
                      transition: 'background-color 0.2s'
                    }}
                    onClick={() => setSelectedClass(parseInt(id))}
                    onMouseEnter={(e) => {
                      if (selectedClass !== parseInt(id)) {
                        e.currentTarget.style.backgroundColor = '#f5f5f5';
                      }
                    }}
                    onMouseLeave={(e) => {
                      if (selectedClass !== parseInt(id)) {
                        e.currentTarget.style.backgroundColor = 'transparent';
                      }
                    }}
                  >
                    <div
                      style={{
                        width: 16,
                        height: 16,
                        backgroundColor: `rgb(${colors[parseInt(id)]?.join(',') || '0,0,0'})`,
                        marginRight: 8,
                        border: '1px solid #ccc',
                        borderRadius: '2px'
                      }}
                    />
                    <span style={{ fontSize: '14px' }}>{name}</span>
                  </div>
                ))}
              </div>
            </div>

            <Space direction="vertical" style={{ width: '100%' }}>
              <Select
                value={selectedModel}
                onChange={setSelectedModel}
                style={{ marginBottom: 10, width: '100%' }}
              >
                <Option value="deeplab">DeepLabV3+</Option>
                <Option value="sam">SAM (原模型)</Option>
                <Option value="sam_finetuned">SAM (微调模型)</Option>
              </Select>
              <Button
                type="primary"
                icon={<RobotOutlined />}
                onClick={handleAIPredict}
                loading={aiLoading}
                disabled={!selectedImage}
              >
                AI辅助标注
              </Button>
              
              <Button
                icon={<SaveOutlined />}
                onClick={handleSaveAnnotation}
                disabled={!selectedImage}
              >
                保存标注
              </Button>
              
              <Button
                onClick={handleTrainModel}
                loading={loading}
              >
                训练模型
              </Button>
              
              <Button
                icon={<EditOutlined />}
                onClick={handleChangeClass}
              >
                修改选中类别
              </Button>
              
              <Button
                icon={<DeleteOutlined />}
                onClick={handleDeleteSelected}
                danger
              >
                删除选中
              </Button>
              
              <Button
                onClick={handleClearAll}
                danger
              >
                清除所有
              </Button>
              
              <Button
                onClick={handleCancelCurrentPolygon}
              >
                取消当前绘制
              </Button>
              
              <Upload
                accept=".yaml,.yml"
                beforeUpload={() => false}
                onChange={handleImportConfig}
                showUploadList={false}
              >
                <Button icon={<ImportOutlined />}>
                  导入配置文件
                </Button>
              </Upload>
            </Space>
          </Card>

          <Card title="图像缩放" size="small">
            <Space direction="vertical" style={{ width: '100%' }}>
              <Button.Group style={{ width: '100%' }}>
                <Button 
                  icon={<ZoomInOutlined />} 
                  onClick={() => canvasRef.current?.zoomIn()}
                  style={{ width: '50%' }}
                >
                  放大
                </Button>
                <Button 
                  icon={<ZoomOutOutlined />} 
                  onClick={() => canvasRef.current?.zoomOut()}
                  style={{ width: '50%' }}
                >
                  缩小
                </Button>
              </Button.Group>
              
              <Button 
                icon={<UndoOutlined />} 
                onClick={() => canvasRef.current?.resetZoom()}
                style={{ width: '100%' }}
              >
                重置缩放
              </Button>
            </Space>
          </Card>

          <Card title="标注信息" size="small">
            <div style={{ fontSize: '12px', color: '#666' }}>
              <div style={{ marginBottom: 8 }}>
                <strong>保存路径:</strong><br />
                {selectedImage ? 
                  `backend/annotations/${selectedImage.filename.replace(/\.[^/.]+$/, '')}.json` : 
                  '请先选择图像'
                }
              </div>
              <div>
                <strong>格式说明:</strong><br />
                JSON格式，包含多边形坐标和类别信息<br />
                <pre style={{ fontSize: '10px', background: '#f5f5f5', padding: '5px', marginTop: '5px' }}>
{`{
  "polygons": [
    {
      "id": "polygon_id",
      "points": [
        {"x": 100, "y": 100},
        {"x": 200, "y": 100}
      ],
      "classId": 1
    }
  ]
}`}
                </pre>
              </div>
            </div>
          </Card>

          <Card title="快捷键" size="small">
            <div style={{ fontSize: '12px', color: '#666' }}>
              <div>Ctrl + 滚轮: 放大/缩小图像</div>
              <div>滚轮: 上下平移图像</div>
              <div>Shift + 滚轮: 左右平移图像</div>
              <div>Ctrl + 拖拽: 平移图像</div>
              <div>点击多边形: 选择类别</div>
              <div>双击: 完成多边形绘制</div>
              <div>取消绘制: 点击"取消当前绘制"按钮</div>
            </div>
          </Card>
        </Space>
      </Sider>
    </Layout>
  );
};

export default MainInterface;