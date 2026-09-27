@echo off
rem ============================================================
rem 修复 VS Code / ZCode Webview "Could not register service worker"
rem 用法：完全退出编辑器后，双击本文件或在本目录运行
rem ============================================================
setlocal
echo.
echo [1/3] 检查编辑器是否在运行...
tasklist | findstr /I "Code.exe ZCode.exe" >nul
if %errorlevel%==0 (
  echo   ⚠ 检测到编辑器正在运行。请先完全退出编辑器（含托盘窗口），再重新运行本脚本。
  pause
  exit /b 1
)
echo   未运行，可以安全清理。

echo [2/3] 清理 Service Worker 缓存...
for %%D in ("%APPDATA%\Code" "%APPDATA%\ZCode") do (
  if exist "%%~D\Service Worker" (
    rmdir /S /Q "%%~D\Service Worker\CacheStorage" 2>nul
    rmdir /S /Q "%%~D\Service Worker\ScriptCache" 2>nul
    rmdir /S /Q "%%~D\Service Worker\Database" 2>nul
    echo   已清理 %%~D\Service Worker
  )
)

echo [3/3] 完成。重新打开编辑器，再试「打开当前章节讲解」。
echo   若仍复现：更新编辑器版本，或到项目 Issues 反馈。
pause
