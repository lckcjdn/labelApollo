import React, { useState } from 'react';
import { Button, Input, message } from 'antd';
import { FolderOpenOutlined } from '@ant-design/icons';
import './DirectoryBrowser.css';

interface FileInfo {
  filename: string;
  filepath: string;
  annotated: boolean;
  annotation_path: string | null;
  file_size: number;
  modified_time: number;
}

interface DirectoryInfo {
  directory_path: string;
  total_files: number;
  annotated_files: number;
  unannotated_files: number;
  files: FileInfo[];
}

interface DirectoryBrowserProps {
  onFileSelect: (filePath: string, annotationPath?: string) => void;
}

const DirectoryBrowser: React.FC<DirectoryBrowserProps> = ({ onFileSelect }) => {
  const [directoryPath, setDirectoryPath] = useState('');
  const [directoryInfo, setDirectoryInfo] = useState<DirectoryInfo | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const selectDirectory = async () => {
    try {
      // 使用 Electron 的目录选择对话框
      if (window.electronAPI && window.electronAPI.selectDirectory) {
        const selectedPath = await window.electronAPI.selectDirectory();
        if (selectedPath) {
          setDirectoryPath(selectedPath);
          setError('');
          // 自动浏览选中的目录
          await browseDirectoryPath(selectedPath);
        }
      } else {
        // 如果不在 Electron 环境中，提供一些常用目录的快捷按钮
        message.info('请手动输入目录路径，或使用快捷目录按钮');
      }
    } catch (err) {
      setError('选择目录失败');
    }
  };

  const selectCommonDirectory = (path: string) => {
    setDirectoryPath(path);
    setError('');
    browseDirectoryPath(path);
  };

  const browseDirectoryPath = async (path: string) => {
    if (!path.trim()) {
      setError('请输入目录路径');
      return;
    }

    setLoading(true);
    setError('');

    try {
      const response = await fetch('/api/browse_directory', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          directory_path: path.trim()
        })
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || '浏览目录失败');
      }

      setDirectoryInfo(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : '浏览目录失败');
      setDirectoryInfo(null);
    } finally {
      setLoading(false);
    }
  };

  const handleFileClick = async (file: FileInfo) => {
    try {
      // 加载图像
      const imageResponse = await fetch('/api/load_image_from_directory', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          image_path: file.filepath
        })
      });

      const imageData = await imageResponse.json();

      if (!imageResponse.ok) {
        throw new Error(imageData.error || '加载图像失败');
      }

      // 如果有标注文件，也加载标注
      let annotationPath = file.annotation_path;
      if (file.annotated && annotationPath) {
        try {
          const annotationResponse = await fetch('/api/load_annotation_from_directory', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({
              image_path: file.filepath,
              annotation_path: annotationPath
            })
          });

          const annotationData = await annotationResponse.json();
          
          if (annotationResponse.ok && annotationData.annotation) {
            // 标注数据已加载，通知父组件
            onFileSelect(imageData.filename, annotationPath);
            return;
          }
        } catch (err) {
          console.warn('加载标注文件失败:', err);
        }
      }

      // 只加载图像
      onFileSelect(imageData.filename);
      
    } catch (err) {
      setError(err instanceof Error ? err.message : '加载文件失败');
    }
  };

  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const formatDate = (timestamp: number): string => {
    return new Date(timestamp * 1000).toLocaleString('zh-CN');
  };

  return (
    <div className="directory-browser">
      <h3>目录浏览器</h3>
      
      <div className="directory-input">
        <Input
          value={directoryPath}
          onChange={(e) => setDirectoryPath(e.target.value)}
          placeholder="输入目录路径 (例如: D:/Images) 或点击选择目录"
          className="directory-path-input"
          suffix={
            <Button
              type="text"
              icon={<FolderOpenOutlined />}
              onClick={selectDirectory}
              title="选择目录"
            />
          }
        />
        <Button 
          type="primary"
          onClick={() => browseDirectoryPath(directoryPath)}
          disabled={loading}
          className="browse-button"
        >
          {loading ? '浏览中...' : '浏览目录'}
        </Button>
      </div>

      {/* 常用目录快捷按钮 */}
      <div className="quick-directories">
        <h4>常用目录</h4>
        <div className="quick-dir-buttons">
          <Button size="small" onClick={() => selectCommonDirectory('D:/')}>D盘</Button>
          <Button size="small" onClick={() => selectCommonDirectory('E:/')}>E盘</Button>
          <Button size="small" onClick={() => selectCommonDirectory('F:/')}>F盘</Button>
          <Button size="small" onClick={() => selectCommonDirectory('C:/Users')}>用户目录</Button>
          <Button size="small" onClick={() => selectCommonDirectory('C:/Users/Public')}>公共目录</Button>
          <Button size="small" onClick={() => selectCommonDirectory('D:/Pictures')}>图片目录</Button>
          <Button size="small" onClick={() => selectCommonDirectory('D:/Downloads')}>下载目录</Button>
          <Button size="small" onClick={() => selectCommonDirectory('D:/Desktop')}>桌面目录</Button>
        </div>
      </div>

      {error && (
        <div className="error-message">
          {error}
        </div>
      )}

      {directoryInfo && (
        <div className="directory-content">
          <div className="directory-stats">
            <h4>目录: {directoryInfo.directory_path}</h4>
            <div className="stats">
              <span className="stat-item">
                总文件: <strong>{directoryInfo.total_files}</strong>
              </span>
              <span className="stat-item annotated">
                已标注: <strong>{directoryInfo.annotated_files}</strong>
              </span>
              <span className="stat-item unannotated">
                未标注: <strong>{directoryInfo.unannotated_files}</strong>
              </span>
            </div>
          </div>

          <div className="file-list">
            <h4>文件列表</h4>
            {directoryInfo.files.length === 0 ? (
              <p className="no-files">该目录中没有找到支持的图像文件</p>
            ) : (
              <div className="files">
                {directoryInfo.files.map((file, index) => (
                  <div
                    key={index}
                    className={`file-item ${file.annotated ? 'annotated' : 'unannotated'}`}
                    onClick={() => handleFileClick(file)}
                  >
                    <div className="file-info">
                      <div className="file-name">
                        {file.filename}
                      </div>
                      <div className="file-details">
                        <span className="file-size">{formatFileSize(file.file_size)}</span>
                        <span className="file-date">{formatDate(file.modified_time)}</span>
                      </div>
                    </div>
                    <div className="file-status">
                      {file.annotated ? (
                        <span className="status-badge annotated">已标注</span>
                      ) : (
                        <span className="status-badge unannotated">未标注</span>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default DirectoryBrowser;