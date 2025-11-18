@echo off
echo ========================================
echo       Label Apollo 停止脚本
echo ========================================
echo.

echo [1/3] 停止后端服务...
taskkill /f /im python.exe /fi "windowtitle eq Label Apollo Backend*" 2>nul
if %errorlevel% equ 0 (
    echo ✅ 后端服务已停止
) else (
    echo ⚠️  未找到运行中的后端服务
)

echo.
echo [2/3] 停止前端服务...
taskkill /f /im node.exe /fi "windowtitle eq Label Apollo Frontend*" 2>nul
if %errorlevel% equ 0 (
    echo ✅ 前端服务已停止
) else (
    echo ⚠️  未找到运行中的前端服务
)

echo.
echo [3/3] 清理端口占用...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :3000') do (
    taskkill /f /pid %%a 2>nul
)
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :5000') do (
    taskkill /f /pid %%a 2>nul
)
echo ✅ 端口清理完成

echo.
echo ========================================
echo           🛑 停止完成！
echo ========================================
echo.
echo 所有Label Apollo相关服务已停止
echo.
pause