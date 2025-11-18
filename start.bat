@echo off
echo ========================================
echo       Label Apollo 启动脚本
echo ========================================
echo.

:: 检查Python环境
echo [1/5] 检查Python环境...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ 错误: 未找到Python，请先安装Python
    pause
    exit /b 1
)
echo ✅ Python环境检查通过

:: 检查后端依赖
echo.
echo [2/5] 检查后端依赖...
cd /d "%~dp0backend"
if not exist "requirements.txt" (
    echo ❌ 错误: 未找到requirements.txt文件
    pause
    exit /b 1
)

:: 安装依赖（如果需要）
echo 正在检查依赖...
pip show torch >nul 2>&1
if %errorlevel% neq 0 (
    echo 正在安装依赖包...
    pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo ❌ 依赖安装失败
        pause
        exit /b 1
    )
)
echo ✅ 后端依赖检查通过

:: 检查模型文件
echo.
echo [3/5] 检查SAM模型文件...
if not exist "models\sam_vit_l.pth" (
    echo ⚠️  未找到SAM模型文件，正在下载...
    python download_sam_model.py
    if %errorlevel% neq 0 (
        echo ❌ 模型下载失败
        pause
        exit /b 1
    )
)
echo ✅ SAM模型文件检查通过

:: 启动后端服务
echo.
echo [4/5] 启动后端服务...
start "Label Apollo Backend" python app.py
timeout /t 3 /nobreak >nul

:: 检查后端是否启动成功
echo 检查后端服务状态...
curl -s http://localhost:5000/api/health >nul 2>&1
if %errorlevel% neq 0 (
    echo ⚠️  后端服务可能还在启动中，请稍等...
    timeout /t 5 /nobreak >nul
)
echo ✅ 后端服务启动成功

:: 启动前端服务
echo.
echo [5/5] 启动前端服务...
cd /d "%~dp0frontend"
if not exist "package.json" (
    echo ❌ 错误: 未找到前端package.json文件
    pause
    exit /b 1
)

:: 检查node_modules
if not exist "node_modules" (
    echo 正在安装前端依赖...
    npm install
    if %errorlevel% neq 0 (
        echo ❌ 前端依赖安装失败
        pause
        exit /b 1
    )
)

start "Label Apollo Frontend" npm start

:: 等待服务完全启动
echo.
echo 等待服务完全启动...
timeout /t 10 /nobreak >nul

:: 显示启动信息
echo.
echo ========================================
echo           🎉 启动完成！
echo ========================================
echo.
echo 📱 前端地址: http://localhost:3000
echo 🔧 后端地址: http://localhost:5000
echo 📖 API文档: http://localhost:5000/api/health
echo.
echo 功能说明:
echo • SAM模型分割标注
echo • 模型微调训练
echo • 批量数据处理
echo • 实时预览效果
echo.
echo 按任意键关闭此窗口...
echo 注意: 关闭此窗口不会停止服务
echo ========================================
pause >nul