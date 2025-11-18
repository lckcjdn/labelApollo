@echo off
echo ========================================
echo       Label Apollo 功能测试
echo ========================================
echo.

echo [1/4] 测试后端健康状态...
cd /d "%~dp0backend"
curl -s http://localhost:5000/api/health
if %errorlevel% equ 0 (
    echo ✅ 后端服务正常
) else (
    echo ❌ 后端服务未响应，请先启动服务
    pause
    exit /b 1
)

echo.
echo [2/4] 测试SAM模型初始化...
python -c "
import requests
try:
    response = requests.post('http://localhost:5000/api/sam_init', 
                           json={'model_type': 'vit_l'}, 
                           timeout=30)
    if response.status_code == 200:
        print('✅ SAM模型初始化成功')
    else:
        print(f'❌ SAM模型初始化失败: {response.status_code}')
except Exception as e:
    print(f'❌ SAM模型初始化异常: {e}')
"

echo.
echo [3/4] 测试前端服务...
curl -s http://localhost:3000 >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ 前端服务正常
) else (
    echo ❌ 前端服务未响应
)

echo.
echo [4/4] 运行基础功能测试...
python test_sam_basic.py

echo.
echo ========================================
echo           📊 测试完成！
echo ========================================
echo.
echo 如果所有测试都通过，说明系统运行正常
echo 可以开始使用Label Apollo进行标注工作
echo.
pause