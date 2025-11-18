import React, { useRef, useEffect, useState, forwardRef, useImperativeHandle, useCallback } from 'react';
import { fabric } from 'fabric';
import { message } from 'antd';

interface Point {
  x: number;
  y: number;
}

interface Polygon {
  id: string;
  points: Point[];
  classId: number;
  fabricObject?: fabric.Polygon;
}

interface AnnotationCanvasProps {
  imagePath: string;
  selectedClass: number;
  colors: { [key: number]: number[] };
  onPolygonClick?: (polygon: Polygon) => void;
}

interface AnnotationCanvasRef {
  loadAnnotation: (annotationData: string) => void;
  loadPrediction: (predictionData: string) => void;
  getAnnotationData: () => string;
  clearAnnotations: () => void;
  deleteSelectedPolygon: () => void;
  changePolygonClass: (classId: number) => void;
  clearCurrentPolygon: () => void;
};
const AnnotationCanvas = forwardRef<AnnotationCanvasRef, AnnotationCanvasProps>(
  ({ imagePath, selectedClass, colors, onPolygonClick }, ref) => {
    // 创建refs用于在事件处理函数中访问最新状态
    const currentPointsRef = useRef<Point[]>([]);
    const tempLineRef = useRef<fabric.Line | null>(null);
    const tempPolygonRef = useRef<fabric.Polygon | null>(null);
    const imageLoadedRef = useRef<boolean>(false);
    const selectedClassRef = useRef<number>(0);
    const colorsRef = useRef<{ [key: number]: number[] }>({});

    const canvasRef = useRef<HTMLCanvasElement>(null);
    const fabricCanvasRef = useRef<fabric.Canvas | null>(null);
    const [currentPoints, setCurrentPoints] = useState<Point[]>([]);
    const [polygons, setPolygons] = useState<Polygon[]>([]);
    const [tempLine, setTempLine] = useState<fabric.Line | null>(null);
    const [tempPolygon, setTempPolygon] = useState<fabric.Polygon | null>(null);
    const originalImageRef = useRef<fabric.Image | null>(null);
  
    // 删除了不再使用的isPanning和lastPanPoint状态
    const [imageLoaded, setImageLoaded] = useState(false);

    // 更新refs当状态变化时
    useEffect(() => { currentPointsRef.current = currentPoints; }, [currentPoints]);
    useEffect(() => { tempLineRef.current = tempLine; }, [tempLine]);
    useEffect(() => { tempPolygonRef.current = tempPolygon; }, [tempPolygon]);
    useEffect(() => { imageLoadedRef.current = imageLoaded; }, [imageLoaded]);
    useEffect(() => { selectedClassRef.current = selectedClass; }, [selectedClass]);
    useEffect(() => { colorsRef.current = colors; }, [colors]);

    // 确保图像始终在底层的辅助函数
    const ensureImageInBackground = useCallback(() => {
      if (fabricCanvasRef.current && originalImageRef.current) {
        const canvas = fabricCanvasRef.current;
        const img = originalImageRef.current;
        
        // 安全检查：确保图像存在于画布对象列表中
        const objects = canvas.getObjects();
        const imageExists = objects.some(obj => obj === img);
        
        if (!imageExists) {
          // 如果图像不存在，重新添加它
          canvas.add(img);
        }
        
        // 重新排序：所有多边形和临时对象在前面，图像在最后面
        objects.forEach(obj => {
          if (obj !== img) {
            canvas.bringToFront(obj);
          }
        });
        
        // 确保图像在最后面
        canvas.sendToBack(img);
        
        // 确保图像属性正确（始终可见、不可交互）
        img.set({
          selectable: false,
          evented: false,
          lockMovementX: true,
          lockMovementY: true,
          lockScalingX: true,
          lockScalingY: true,
          lockRotation: true,
          hasControls: false,
          hasBorders: false,
          visible: true  // 确保图像始终可见，避免被隐藏
        });
        
        // 重新渲染
        canvas.renderAll();
      }
    }, []);




        // 添加唯一标识
        const polygonId = `polygon_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
        (polygon as any).polygonId = polygonId;

        // 清理临时元素
        if (tempLine) {
          fabricCanvasRef.current?.remove(tempLine);
          setTempLine(null);
        }
        if (tempPolygon) {
          fabricCanvasRef.current?.remove(tempPolygon);
          setTempPolygon(null);
        }

        // 清理点标记
        const objects = fabricCanvasRef.current?.getObjects() || [];
        objects.forEach(obj => {
          if (obj instanceof fabric.Circle && obj.radius === 3) {
            fabricCanvasRef.current?.remove(obj);
          }
        });

        // 添加多边形到画布
        fabricCanvasRef.current?.add(polygon);
        
        // 确保正确的层级顺序
        ensureImageInBackground();
        
        fabricCanvasRef.current?.renderAll();
        
        // 更新状态，将点转换为相对于原图的坐标
        const imgScale = img.scaleX || 1;
        const newPolygon: Polygon = {
          id: polygonId,
          points: validPoints.map(p => ({ 
            x: (p.x - imgLeft) / imgScale, 
            y: (p.y - imgTop) / imgScale 
          })),
          classId: selectedClass
        };
        
        setPolygons([...polygons, newPolygon]);
        setCurrentPoints([]);
        
        message.success('多边形创建成功');
      } catch (error) {
        console.error('创建多边形失败:', error);
        message.error('创建多边形失败');
      }
    }, [currentPoints, selectedClass, colors, polygons, tempLine, tempPolygon, ensureImageInBackground]);

    // 删除了zoomIn函数

    // 删除了zoomOut函数

    // 删除了resetZoom函数

    // 初始化画布
    useEffect(() => {
      if (canvasRef.current) {
        const canvas = new fabric.Canvas(canvasRef.current, {
          width: 800,
          height: 600,
          backgroundColor: '#f0f0f0',
          selection: false
        });
        
        fabricCanvasRef.current = canvas;

        // 禁用默认的绘图模式
        canvas.isDrawingMode = false;

        // 事件处理函数可以直接使用顶层声明的refs来访问最新状态

        // 处理鼠标按下事件
        const handleMouseDown = (opt: fabric.IEvent) => {
          if (!fabricCanvasRef.current || !originalImageRef.current || !imageLoadedRef.current) return;
          
          const pointer = fabricCanvasRef.current.getPointer(opt.e);
          
          // 检查是否点击在图像范围内
          const img = originalImageRef.current;
          const imgLeft = img.left || 0;
          const imgTop = img.top || 0;
          const imgWidth = (img.width || 0) * (img.scaleX || 1);
          const imgHeight = (img.height || 0) * (img.scaleY || 1);
          
          if (pointer.x < imgLeft || pointer.x > imgLeft + imgWidth ||
              pointer.y < imgTop || pointer.y > imgTop + imgHeight) {
            return;
          }
          
          // 添加点
          const newPoint = new fabric.Point(pointer.x, pointer.y);
          const updatedPoints = [...currentPointsRef.current, newPoint];
          setCurrentPoints(updatedPoints);
          
          // 创建点标记
          const circle = new fabric.Circle({
            left: pointer.x - 3,
            top: pointer.y - 3,
            radius: 3,
            fill: 'red',
            stroke: 'white',
            strokeWidth: 1,
            selectable: false,
            evented: false,
            hasBorders: false,
            hasControls: false
          });
          
          fabricCanvasRef.current.add(circle);
          
          // 创建临时线条
          if (tempLineRef.current) {
            fabricCanvasRef.current.remove(tempLineRef.current);
          }
          
          if (updatedPoints.length > 1) {
            const line = new fabric.Line([
              updatedPoints[updatedPoints.length - 2].x,
              updatedPoints[updatedPoints.length - 2].y,
              pointer.x,
              pointer.y
            ], {
              stroke: 'rgba(255, 0, 0, 0.5)',
              strokeWidth: 2,
              selectable: false,
              evented: false,
              hasBorders: false,
              hasControls: false
            });
            
            fabricCanvasRef.current.add(line);
            setTempLine(line);
          }
          
          // 创建临时多边形
          if (tempPolygonRef.current) {
            fabricCanvasRef.current.remove(tempPolygonRef.current);
          }
          
          if (updatedPoints.length > 2) {
            const polygon = new fabric.Polygon(updatedPoints, {
              fill: 'rgba(255, 0, 0, 0.1)',
              stroke: 'rgba(255, 0, 0, 0.5)',
              strokeWidth: 2,
              selectable: false,
              evented: false,
              hasBorders: false,
              hasControls: false
            });
            
            fabricCanvasRef.current.add(polygon);
            setTempPolygon(polygon);
          }
          
          ensureImageInBackground();
          fabricCanvasRef.current.renderAll();
        };

        // 处理鼠标移动事件
        const handleMouseMove = (opt: fabric.IEvent) => {
          if (!fabricCanvasRef.current || !originalImageRef.current) return;
          
          const pointer = fabricCanvasRef.current.getPointer(opt.e);
          
          // 更新临时线条
          if (currentPointsRef.current.length > 0 && tempLineRef.current) {
            tempLineRef.current.set({ x2: pointer.x, y2: pointer.y });
            
            // 更新临时多边形
            if (tempPolygonRef.current && currentPointsRef.current.length >= 2) {
              const tempPoints = [...currentPointsRef.current, { x: pointer.x, y: pointer.y }];
              tempPolygonRef.current.set({ points: tempPoints.map(p => new fabric.Point(p.x, p.y)) });
            }
            
            fabricCanvasRef.current.renderAll();
            ensureImageInBackground();
          }
        };

        // 处理鼠标释放事件
        const handleMouseUp = () => {
          // 空函数，已删除平移功能
        };

        // 处理双击事件 - 完成多边形绘制
        const handleDoubleClick = (e: fabric.IEvent) => {
          if (currentPointsRef.current.length < 3) {
            message.warning('至少需要3个点才能创建多边形');
            return;
          }

          try {
            // 获取图像的实际显示区域
            const img = originalImageRef.current;
            if (!img) {
              message.error('图像未加载，无法创建标注');
              return;
            }
            
            const imgLeft = img.left || 0;
            const imgTop = img.top || 0;
            const imgWidth = (img.width || 0) * (img.scaleX || 1);
            const imgHeight = (img.height || 0) * (img.scaleY || 1);

            // 验证并限制所有点在图像边界内
            const validPoints = currentPointsRef.current.map(point => {
              const x = Math.max(imgLeft, Math.min(imgLeft + imgWidth, point.x));
              const y = Math.max(imgTop, Math.min(imgTop + imgHeight, point.y));
              return { x, y };
            });

            // 从图像坐标系转换为相对坐标系
            const polygonPoints = validPoints.map(point => {
              const relativeX = (point.x - imgLeft) / imgWidth;
              const relativeY = (point.y - imgTop) / imgHeight;
              return { x: relativeX, y: relativeY };
            });

            // 创建新的多边形对象
            const color = colorsRef.current[selectedClassRef.current] || [255, 0, 0];
            const rgbaColor = `rgba(${color[0]}, ${color[1]}, ${color[2]}, 0.3)`;
            const strokeColor = `rgb(${color[0]}, ${color[1]}, ${color[2]})`;

            const fabricPolygon = new fabric.Polygon(validPoints, {
              fill: rgbaColor,
              stroke: strokeColor,
              strokeWidth: 2,
              selectable: true,
              evented: true,
              hasBorders: true,
              hasControls: true,
              lockScalingX: true,
              lockScalingY: true,
              lockRotation: true
            });

            if (fabricCanvasRef.current) {
                const canvas = fabricCanvasRef.current;
                canvas.add(fabricPolygon);

                // 创建多边形数据对象
                const newPolygon: Polygon = {
                  id: Math.random().toString(36).substring(2, 15) + Math.random().toString(36).substring(2, 15),
                  points: polygonPoints,
                  classId: selectedClassRef.current,
                  fabricObject: fabricPolygon
                };

                // 添加多边形到状态
                setPolygons(prevPolygons => [...prevPolygons, newPolygon]);

                // 重置临时状态
                setCurrentPoints([]);
                if (tempLineRef.current) {
                  canvas.remove(tempLineRef.current);
                  setTempLine(null);
                }
                if (tempPolygonRef.current) {
                  canvas.remove(tempPolygonRef.current);
                  setTempPolygon(null);
                }

                // 清除所有点标记
                canvas.getObjects('circle').forEach(circle => {
                  canvas.remove(circle);
                });

                // 确保图像在背景层
                ensureImageInBackground();
                canvas.renderAll();

                message.success('多边形创建成功');
            }
          } catch (error) {
            console.error('创建多边形失败:', error);
            message.error('创建多边形失败');
          }
        };

        // 添加事件监听器
        canvas.on('mouse:down', handleMouseDown);
        canvas.on('mouse:move', handleMouseMove);
        canvas.on('mouse:up', handleMouseUp);
        canvas.on('mouse:dblclick', handleDoubleClick);
        return () => {
          canvas.dispose();
        };
      }
    }, [ensureImageInBackground]);

    // 加载图像
    const loadImage = useCallback(async () => {
      if (!fabricCanvasRef.current || !imagePath) return;

      // 开始加载图像时重置状态
      setImageLoaded(false);
      
      try {
        // 构建正确的图像URL
        let imageUrl: string;
        if (imagePath.startsWith('http')) {
          imageUrl = imagePath;
        } else {
          // 确保路径格式正确
          const cleanImagePath = imagePath.replace(/^\/+/, ''); // 移除开头的斜杠
          imageUrl = `http://localhost:5000/uploads/${cleanImagePath.replace(/^uploads\//, '')}`;
        }

        const img = await new Promise<HTMLImageElement>((resolve, reject) => {
          const img = new Image();
          img.onload = () => resolve(img);
          img.onerror = () => reject(new Error(`Failed to load image: ${imageUrl}`));
          img.crossOrigin = 'anonymous'; // 解决跨域问题
          img.src = imageUrl;
        });

        const canvas = fabricCanvasRef.current;

        // 获取容器尺寸
        const container = canvas.getElement().parentElement as HTMLElement;
        if (!container) {
          console.error('Canvas container not found');
          // 设置图像加载失败状态
          setImageLoaded(false);
          message.error('Canvas container not found');
          return;
        }
        const containerWidth = container.offsetWidth;
        const containerHeight = container.offsetHeight;

        // 获取图像原始尺寸
        const imgWidth = img.width;
        const imgHeight = img.height;

        // 计算缩放比例，使图像完全显示在容器内
        const scaleX = containerWidth / imgWidth;
        const scaleY = containerHeight / imgHeight;
        const scale = Math.min(scaleX, scaleY);

        // 设置画布尺寸为容器尺寸
        canvas.setDimensions({
          width: containerWidth,
          height: containerHeight
        });

        // 清除现有图像
        if (originalImageRef.current) {
          canvas.remove(originalImageRef.current);
          originalImageRef.current = null;
        }

        // 创建图像对象并添加到画布，居中显示
        const fabricImg = new fabric.Image(img, {
          left: (containerWidth - imgWidth * scale) / 2,
          top: (containerHeight - imgHeight * scale) / 2,
          scaleX: scale,
          scaleY: scale,
          selectable: false,
          evented: false,
          // 确保图像不会被意外修改
          lockMovementX: true,
          lockMovementY: true,
          lockScalingX: true,
          lockScalingY: true,
          lockRotation: true,
          hasControls: false,
          hasBorders: false
        });

        canvas.add(fabricImg);
        originalImageRef.current = fabricImg;
        
        // 确保图像在最底层
        ensureImageInBackground();
        
        // 强制渲染
        canvas.renderAll();
        
        // 设置图像加载完成状态
        setImageLoaded(true);
        
        console.log('Canvas setup completed');
      } catch (error) {
        console.error('加载图像失败:', error);
        // 设置图像加载失败状态
        setImageLoaded(false);
        // 显示错误消息给用户
        message.error(`图像加载失败: ${error instanceof Error ? error.message : '未知错误'}`);
      }
    }, [imagePath, ensureImageInBackground]);

    // 加载图像
    useEffect(() => {
      loadImage();
    }, [loadImage]);

    // 清除所有标注
    const clearAnnotations = useCallback(() => {
      if (!fabricCanvasRef.current) return;
      
      const canvas = fabricCanvasRef.current;
      const img = originalImageRef.current;
      
      // 安全地清除所有非图像对象
      const objectsToRemove: fabric.Object[] = [];
      const objects = canvas.getObjects();
      
      objects.forEach(obj => {
        if (obj !== img) {
          objectsToRemove.push(obj);
        }
      });
      
      // 批量删除以提高性能
      objectsToRemove.forEach(obj => canvas.remove(obj));
      
      // 重置状态
      setPolygons([]);
      setCurrentPoints([]);
      setTempLine(null);
      setTempPolygon(null);
      
      // 确保图像仍然在正确的位置和状态
      if (img) {
        ensureImageInBackground();
      }
    }, [ensureImageInBackground]);

    // 删除选中的多边形
    const deleteSelectedPolygon = useCallback(() => {
      if (!fabricCanvasRef.current) return;
      
      const activeObject = fabricCanvasRef.current.getActiveObject();
      if (activeObject && activeObject instanceof fabric.Polygon) {
        const polygonId = (activeObject as any).polygonId;
        fabricCanvasRef.current.remove(activeObject);
        setPolygons(prevPolygons => prevPolygons.filter(p => p.id !== polygonId));
      }
    }, []);

    // 修改多边形类别
    const changePolygonClass = useCallback((classId: number) => {
      if (!fabricCanvasRef.current) return;
      
      const canvas = fabricCanvasRef.current;
      const activeObject = canvas.getActiveObject();
      if (activeObject && activeObject instanceof fabric.Polygon) {
        const newColor = `rgba(${colors[classId]?.join(',') || '0,0,0'}, 0.5)`;
        const newStrokeColor = `rgb(${colors[classId]?.join(',') || '0,0,0'})`;
        
        activeObject.set({
          fill: newColor,
          stroke: newStrokeColor
        });
        
        canvas.renderAll();
        
        // 确保图像在背景层
        ensureImageInBackground();
        
        // 更新多边形数据
        const polygonId = (activeObject as any).polygonId;
        setPolygons(prevPolygons => prevPolygons.map(p => 
          p.id === polygonId ? { ...p, classId } : p
        ));
      }
    }, [colors, ensureImageInBackground]);

    // 加载标注数据
    const loadAnnotation = useCallback((annotationData: string) => {
      if (!fabricCanvasRef.current || !originalImageRef.current) return;
      
      try {
        const canvas = fabricCanvasRef.current;
        const img = originalImageRef.current;
        const annotation = JSON.parse(annotationData);
        
        // 清除现有标注
        const objectsToRemove: fabric.Object[] = [];
        canvas.getObjects().forEach(obj => {
          if (obj !== img) {
            objectsToRemove.push(obj);
          }
        });
        
        objectsToRemove.forEach(obj => canvas.remove(obj));
        
        // 创建新多边形
        const newPolygons: Polygon[] = [];
        
        for (const polyData of annotation.polygons) {
          const { points, classId, id } = polyData;
          
          // 缩放坐标
          const imgLeft = img.left || 0;
          const imgTop = img.top || 0;
          const imgScale = img.scaleX || 1;
          
          const scaledPoints = points.map((p: Point) => ({ x: imgLeft + p.x * imgScale, y: imgTop + p.y * imgScale }));
          const fabricPoints = scaledPoints.map((p: { x: number; y: number }) => new fabric.Point(p.x, p.y));
          
          const color = colors[classId] || [255, 0, 0];
          const polygon = new fabric.Polygon(fabricPoints, {
            fill: `rgba(${color[0]}, ${color[1]}, ${color[2]}, 0.5)`,
            stroke: `rgb(${color[0]}, ${color[1]}, ${color[2]})`,
            strokeWidth: 2,
            selectable: true,
            evented: true,
            objectCaching: false,
            // 限制标注操作
            lockMovementX: true,
            lockMovementY: true,
            lockScalingX: true,
            lockScalingY: true,
            lockRotation: true,
            lockSkewingX: true,
            lockSkewingY: true
          } as any);
          
          // 手动添加polygonId属性
          (polygon as any).polygonId = id;
          
          // 添加到画布
          canvas.add(polygon);
          
          // 保存到状态
          newPolygons.push({ id, points, classId, fabricObject: polygon });
        }
        
        setPolygons(newPolygons);
        setCurrentPoints([]);
        setTempLine(null);
        setTempPolygon(null);
        
        // 确保正确的层级顺序
        ensureImageInBackground();
        
        canvas.renderAll();
        message.success('标注数据加载成功');
      } catch (error) {
        console.error('加载标注数据失败:', error);
        message.error('标注数据格式错误');
      }
    }, [colors, ensureImageInBackground]);

    // 加载预测数据
    const loadPrediction = useCallback((predictionData: string) => {
      // 类似于loadAnnotation，但使用不同的样式
      loadAnnotation(predictionData);
    }, [loadAnnotation]);

    // 获取标注数据
    const getAnnotationData = useCallback(() => {
      const annotationData = {
        polygons: polygons.map(p => ({
          id: p.id,
          points: p.points,
          classId: p.classId
        })),
        imagePath: imagePath,
        timestamp: new Date().toISOString()
      };
      
      return JSON.stringify(annotationData);
    }, [polygons, imagePath]);

    const clearCurrentPolygon = useCallback(() => {
      // 清除当前正在绘制的多边形
      setCurrentPoints([]);
      
      // 清除临时线条和多边形
      if (tempLine && fabricCanvasRef.current) {
        fabricCanvasRef.current.remove(tempLine);
        setTempLine(null);
      }
      
      if (tempPolygon && fabricCanvasRef.current) {
        fabricCanvasRef.current.remove(tempPolygon);
        setTempPolygon(null);
      }
      
      // 清除所有点标记
      if (fabricCanvasRef.current) {
        const objects = fabricCanvasRef.current.getObjects();
        objects.forEach(obj => {
          if (obj.type === 'circle' && !obj.selectable) {
            fabricCanvasRef.current!.remove(obj);
          }
        });
        fabricCanvasRef.current.renderAll();
      }
    }, [tempLine, tempPolygon]);

    // 暴露方法给父组件
    useImperativeHandle(ref, () => ({
      clearAnnotations,
      deleteSelectedPolygon,
      changePolygonClass,
      loadAnnotation,
      loadPrediction,
      getAnnotationData,
      clearCurrentPolygon
    }), [clearAnnotations, deleteSelectedPolygon, changePolygonClass, loadAnnotation, loadPrediction, getAnnotationData, clearCurrentPolygon]);

    return (
      <div style={{ position: 'relative', height: '100%' }}>
        <canvas ref={canvasRef} style={{ height: '100%', width: '100%' }} />
        <div style={{ 
          marginTop: 10, 
          fontSize: '12px', 
          color: '#666',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center'
        }}>
          <span>点击添加顶点，双击完成多边形</span>
          <span></span>
        </div>
      </div>
    );
  }
);

export default AnnotationCanvas;